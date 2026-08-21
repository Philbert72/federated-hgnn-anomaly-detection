#!/usr/bin/env python3
"""Generate synthetic AWS CloudTrail logs for the federated HGNN experiments.

Design goal: attackers must be genuinely hard to distinguish from normal users.
A compromised account is modelled as a *normal user who also does something bad* —
it keeps doing its ordinary job, from the ordinary corporate IPs, using ordinary
API calls, with a limited window of malicious activity mixed in.

Every parameter below is set from that premise, not tuned against model output.
Deliberately avoided (all present in the v1 dataset, all of which made the task trivial):

  - attacker volume far outside the normal range      -> volumes now overlap
  - attackers on their own IP subnet                  -> shared corporate egress pool
  - attacker-exclusive API vocabulary                 -> ~90% shared vocabulary
  - normal users who never perform a write            -> normals write routinely
  - one attack scenario cloned N times                -> 4 scenarios, varied intensity

Usage:
    python3 Dataset/generate_synthetic_logs.py OUTPUT_DIR [--users 300] [--seed 42]
"""
import argparse
import json
import os
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np

ACCOUNT_ID = "123837392027"
REGIONS = ["us-east-1", "us-east-1", "us-east-1", "us-west-2", "eu-west-1"]

# ---------------------------------------------------------------------------
# API catalogue. Attackers draw from the same pools as everyone else; only the
# mixture and the rate differ.
# ---------------------------------------------------------------------------
READ_EVENTS = {
    "s3.amazonaws.com": ["GetObject", "ListBuckets", "HeadObject", "GetBucketPolicy",
                         "ListObjects", "GetBucketAcl", "GetBucketLocation"],
    "iam.amazonaws.com": ["GetUser", "ListRoles", "ListUsers", "GetRole",
                          "ListAttachedRolePolicies", "ListAccessKeys", "GetPolicy"],
    "ec2.amazonaws.com": ["DescribeInstances", "DescribeSecurityGroups", "DescribeVpcs",
                          "DescribeSubnets", "DescribeVolumes", "DescribeImages"],
    "kms.amazonaws.com": ["Decrypt", "DescribeKey", "ListKeys", "GetKeyPolicy"],
    "secretsmanager.amazonaws.com": ["GetSecretValue", "DescribeSecret", "ListSecrets"],
    "ssm.amazonaws.com": ["GetParameter", "DescribeParameters", "GetParameters"],
    "rds.amazonaws.com": ["DescribeDBInstances", "DescribeDBSnapshots", "DescribeDBClusters"],
    "logs.amazonaws.com": ["FilterLogEvents", "DescribeLogGroups", "GetLogEvents"],
    "lambda.amazonaws.com": ["GetFunction", "ListFunctions", "GetFunctionConfiguration"],
    "dynamodb.amazonaws.com": ["Query", "Scan", "DescribeTable", "GetItem"],
    "sts.amazonaws.com": ["GetCallerIdentity"],
}

WRITE_EVENTS = {
    "s3.amazonaws.com": ["PutObject", "DeleteObject", "CreateBucket", "PutBucketPolicy"],
    "iam.amazonaws.com": ["TagUser", "UpdateAccessKey", "CreateServiceLinkedRole"],
    "ec2.amazonaws.com": ["CreateTags", "RunInstances", "StopInstances", "CreateSnapshot"],
    "kms.amazonaws.com": ["Encrypt", "GenerateDataKey"],
    "ssm.amazonaws.com": ["PutParameter", "DeleteParameter", "SendCommand"],
    "rds.amazonaws.com": ["CreateDBSnapshot", "ModifyDBInstance"],
    "lambda.amazonaws.com": ["InvokeFunction", "UpdateFunctionCode"],
    "dynamodb.amazonaws.com": ["PutItem", "UpdateItem", "DeleteItem"],
    "logs.amazonaws.com": ["CreateLogStream", "PutLogEvents"],
}

# Sensitive calls. Attackers lean on these, but admins and ops issue them during
# ordinary work too, so their presence alone must never be conclusive.
HIGH_RISK = {
    "sts.amazonaws.com": ["AssumeRole"],
    "iam.amazonaws.com": ["CreateAccessKey", "AttachUserPolicy", "PutUserPolicy",
                          "CreateLoginProfile", "UpdateLoginProfile", "CreateRole"],
    "secretsmanager.amazonaws.com": ["GetSecretValue"],
    "kms.amazonaws.com": ["Decrypt"],
    "s3.amazonaws.com": ["GetObject", "ListBuckets", "PutBucketPolicy"],
}

ROLES = {
    # role: (preferred services, write propensity, high-risk propensity)
    "developer": (["s3.amazonaws.com", "lambda.amazonaws.com", "dynamodb.amazonaws.com",
                   "logs.amazonaws.com", "ssm.amazonaws.com"], 0.35, 0.02),
    "analyst":   (["s3.amazonaws.com", "dynamodb.amazonaws.com", "rds.amazonaws.com",
                   "logs.amazonaws.com"], 0.18, 0.01),
    "ops":       (["ec2.amazonaws.com", "ssm.amazonaws.com", "logs.amazonaws.com",
                   "rds.amazonaws.com", "kms.amazonaws.com"], 0.40, 0.05),
    "admin":     (["iam.amazonaws.com", "ec2.amazonaws.com", "kms.amazonaws.com",
                   "secretsmanager.amazonaws.com", "sts.amazonaws.com"], 0.30, 0.12),
}

# Each scenario tilts the attack burst toward particular services. Intensity is the
# fraction of the account's events that are malicious — deliberately reaching down
# to 0.06 so that some attackers are genuinely near-invisible.
SCENARIOS = {
    "credential_access":    ["secretsmanager.amazonaws.com", "kms.amazonaws.com", "iam.amazonaws.com"],
    "privilege_escalation": ["iam.amazonaws.com", "sts.amazonaws.com"],
    "data_exfiltration":    ["s3.amazonaws.com", "dynamodb.amazonaws.com"],
    "persistence":          ["iam.amazonaws.com", "lambda.amazonaws.com", "ec2.amazonaws.com"],
}

USER_AGENTS = [
    "aws-cli/2.13.0 Python/3.11.4 Linux/5.15.0 exe/x86_64.ubuntu.22",
    "[S3Console/0.4, aws-internal/3 aws-sdk-java/1.12.488 Linux/5.4.247]",
    "Boto3/1.28.3 Python/3.10.12 Linux/5.15.0-76-generic",
    "aws-sdk-go/1.44.261 (go1.19.8; linux; amd64)",
    "console.amazonaws.com",
    "terraform-provider-aws/5.10.0 (+https://registry.terraform.io)",
]

ERROR_CODES = ["AccessDenied", "UnauthorizedOperation", "NoSuchEntity",
               "ThrottlingException", "ValidationException"]


def make_ip_pool(rng, n_shared=12):
    """Corporate egress: VPN pools, office NAT, bastion hosts — shared by many users."""
    pool = [f"10.20.{rng.integers(1, 6)}.{rng.integers(2, 250)}" for _ in range(n_shared // 2)]
    pool += [f"192.168.{rng.integers(1, 4)}.{rng.integers(2, 250)}" for _ in range(n_shared - n_shared // 2)]
    return list(dict.fromkeys(pool))


def pick_events(rng, services, n, write_p, risk_p):
    """Draw n (service, event, readOnly) triples for a given behavioural mix."""
    out = []
    for _ in range(n):
        svc = rng.choice(services)
        roll = rng.random()
        if roll < risk_p and svc in HIGH_RISK:
            ev = rng.choice(HIGH_RISK[svc])
            read_only = ev.startswith(("Get", "List", "Describe")) or ev == "Decrypt"
        elif roll < risk_p + write_p and svc in WRITE_EVENTS:
            ev = rng.choice(WRITE_EVENTS[svc])
            read_only = False
        else:
            ev = rng.choice(READ_EVENTS.get(svc, READ_EVENTS["s3.amazonaws.com"]))
            read_only = True
        out.append((svc, ev, read_only))
    return out


def business_time(rng, start, days):
    """Weekday, business-hours-weighted timestamp."""
    for _ in range(20):
        t = start + timedelta(days=float(rng.uniform(0, days)))
        if t.weekday() < 5 or rng.random() < 0.12:      # occasional weekend work
            hour = int(np.clip(rng.normal(13, 3), 0, 23)) if rng.random() < 0.88 \
                else int(rng.integers(0, 24))
            return t.replace(hour=hour, minute=int(rng.integers(0, 60)),
                             second=int(rng.integers(0, 60)), microsecond=0)
    return start


def build_record(rng, user, ip, svc, event, read_only, ts, is_attack, err_p):
    key_id = user["access_key"]
    rec = {
        "eventVersion": "1.08",
        "userIdentity": {
            "type": "IAMUser",
            "principalId": user["principal_id"],
            "arn": f"arn:aws:iam::{ACCOUNT_ID}:user/{user['name']}",
            "accountId": ACCOUNT_ID,
            "accessKeyId": key_id,
            "userName": user["name"],
            "sessionContext": {
                "sessionIssuer": {},
                "webIdFederationData": {},
                "attributes": {
                    "creationDate": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "mfaAuthenticated": "true" if user["mfa"] else "false",
                },
            },
        },
        "eventTime": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "eventSource": svc,
        "eventName": event,
        "awsRegion": user["region"],
        "sourceIPAddress": ip,
        "userAgent": user["agent"],
        "requestParameters": {"Host": f"{svc}"},
        "responseElements": None,
        "requestID": str(uuid.UUID(bytes=rng.bytes(16))),
        "eventID": str(uuid.UUID(bytes=rng.bytes(16))),
        "readOnly": bool(read_only),
        "eventType": "AwsApiCall",
        "managementEvent": True,
        "recipientAccountId": ACCOUNT_ID,
        "eventCategory": "Management",
        # Ground truth kept per-record so session-level work stays possible.
        "_is_attack": bool(is_attack),
    }
    if rng.random() < err_p:
        rec["errorCode"] = str(rng.choice(ERROR_CODES))
    return rec


def generate(n_users, attacker_frac, seed, days=21):
    rng = np.random.default_rng(seed)
    ip_pool = make_ip_pool(rng)
    start = datetime(2024, 3, 4, tzinfo=timezone.utc)

    n_attackers = int(round(n_users * attacker_frac))
    labels = rng.permutation([1] * n_attackers + [0] * (n_users - n_attackers))

    users, manifest = [], []
    for i, lab in enumerate(labels):
        role = str(rng.choice(list(ROLES)))
        pref, write_p, risk_p = ROLES[role]
        # Volume drawn from ONE distribution for both classes — this is the key
        # property v1 lacked. Attackers get no volume advantage.
        n_events = int(np.clip(rng.lognormal(np.log(140), 0.55), 60, 500))
        users.append({
            "name": f"user_{i:03d}",
            "role": role, "services": pref, "write_p": write_p, "risk_p": risk_p,
            "n_events": n_events, "label": int(lab),
            "principal_id": "AIDA" + "".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"), 17)),
            "access_key": "AKIA" + "".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"), 16)),
            "mfa": bool(rng.random() < 0.75),
            "region": str(rng.choice(REGIONS)),
            "agent": str(rng.choice(USER_AGENTS)),
            # Everyone draws from the same corporate egress pool.
            "ips": list(rng.choice(ip_pool, size=int(rng.integers(1, 4)), replace=False)),
            "err_p": float(rng.uniform(0.01, 0.15)),
        })

    for u in users:
        recs = []
        if u["label"] == 0:
            for svc, ev, ro in pick_events(rng, u["services"], u["n_events"],
                                           u["write_p"], u["risk_p"]):
                ts = business_time(rng, start, days)
                recs.append(build_record(rng, u, str(rng.choice(u["ips"])), svc, ev, ro,
                                         ts, False, u["err_p"]))
            manifest.append({"user": u["name"], "label": 0, "scenario": None,
                             "role": u["role"], "attack_window": None})
        else:
            scenario = str(rng.choice(list(SCENARIOS)))
            intensity = float(rng.uniform(0.06, 0.35))   # some attacks are very faint
            n_attack = max(int(u["n_events"] * intensity), 5)
            n_benign = u["n_events"] - n_attack

            # The compromised account keeps doing its day job.
            for svc, ev, ro in pick_events(rng, u["services"], n_benign,
                                           u["write_p"], u["risk_p"]):
                ts = business_time(rng, start, days)
                recs.append(build_record(rng, u, str(rng.choice(u["ips"])), svc, ev, ro,
                                         ts, False, u["err_p"]))

            # Malicious activity: concentrated in a window, tilted toward the
            # scenario's services, and mostly from the user's usual IPs.
            burst_start = business_time(rng, start, days - 1)
            attack_ips = list(u["ips"])
            if rng.random() < 0.3:                      # sometimes a new device/location
                attack_ips.append(f"10.20.{rng.integers(1, 6)}.{rng.integers(2, 250)}")
            # An intruder works through the access the account already has, pivoting
            # to only one or two new services. Letting the attack use a wholly
            # separate service set would hand the model a free "service count" tell.
            n_pivot = 1 if rng.random() < 0.7 else 2
            pivots = list(rng.choice(SCENARIOS[scenario],
                                     size=min(n_pivot, len(SCENARIOS[scenario])),
                                     replace=False))
            svc_mix = list(u["services"]) + pivots
            for svc, ev, ro in pick_events(rng, svc_mix, n_attack,
                                           write_p=0.30, risk_p=0.45):
                ts = burst_start + timedelta(minutes=float(rng.exponential(45)))
                recs.append(build_record(rng, u, str(rng.choice(attack_ips)), svc, ev, ro,
                                         ts, True, min(u["err_p"] * 1.6, 0.4)))

            manifest.append({
                "user": u["name"], "label": 1, "scenario": scenario, "role": u["role"],
                "attack_window": [burst_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                  (burst_start + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")],
            })

        recs.sort(key=lambda r: r["eventTime"])
        u["records"] = recs

    return users, manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--users", type=int, default=300)
    ap.add_argument("--attacker-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = os.path.expanduser(args.output_dir)
    os.makedirs(out, exist_ok=True)

    users, manifest = generate(args.users, args.attacker_frac, args.seed)
    total = 0
    for u in users:
        with open(os.path.join(out, f"{u['name']}_logs.json"), "w") as fh:
            json.dump({"Records": u["records"]}, fh, indent=1)
        total += len(u["records"])

    with open(os.path.join(out, "labels.json"), "w") as fh:
        json.dump({"entities": manifest}, fh, indent=1)

    n_atk = sum(m["label"] for m in manifest)
    print(f"Wrote {len(users)} users / {total} records to {out}")
    print(f"  anomalous: {n_atk} ({n_atk / len(users):.1%})")
    print(f"  labels.json written (filenames carry no class information)")


if __name__ == "__main__":
    main()
