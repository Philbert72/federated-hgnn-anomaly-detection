# federated-hgnn-anomaly-detection

Privacy-Preserving Detection of Anomalous Identity Access Patterns in Multi-Cloud
Environments using Federated HGNN.

A heterogeneous graph (users, IPs, services) is built from AWS CloudTrail logs and
classified with a HANConv-based model, first centrally as a baseline and then across
three simulated federated clients using FedAvg, so no client shares raw records.

## Getting started

```bash
git clone https://github.com/Philbert72/federated-hgnn-anomaly-detection.git
cd federated-hgnn-anomaly-detection

python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install torch first, matched to your platform: https://pytorch.org
pip install -r requirements.txt
```

Then create the data directory and populate it — see [Datasets](#datasets) below.
It is gitignored, so it does **not** exist in a fresh clone:

```bash
mkdir -p Dataset/Raw/CloudTrail/CloudTrail
```

Finally run the notebooks **in order** (`Phase1 → Phase5`). Each phase writes files
that the next one reads, so running them out of order will fail:

```bash
jupyter notebook Notebooks/
```

Or headless:

```bash
for p in Phase1_Data_Preparation Phase2_Feature_Engineering Phase3_Graph_Construction \
         Phase4_HGNN_Model Phase5_Federated_Learning; do
  python3 -m nbconvert --to notebook --execute --inplace "Notebooks/$p.ipynb"
done
```

## Datasets

Raw logs are **not** committed (`Dataset/Raw/` is gitignored). Every notebook reads
from this exact path — note that `CloudTrail` is nested twice:

```
Dataset/Raw/CloudTrail/CloudTrail/*.json
```

If that folder is empty, Phase 1 fails while flattening `userIdentity`, because the
glob matched nothing. That means missing data, not a bug in the code.

### Option A — generate it (no download, reproducible)

```bash
python3 Dataset/generate_synthetic_logs.py Dataset/Raw/CloudTrail/CloudTrail \
    --users 300 --seed 42
```

Writes 300 accounts (15% compromised) plus a `labels.json` manifest. The seed makes
this byte-for-byte reproducible, so collaborators regenerate identical data rather
than passing files around.

Attackers are modelled as compromised accounts that keep doing ordinary work:
event volume overlapping normal users, the same corporate egress IPs, ~90% shared
API vocabulary, and four attack scenarios at varied intensity. This matters —
earlier synthetic data separated the classes by ~77 standard deviations on event
volume alone, so a one-line threshold rule scored F1 = 1.00 and the model result
meant nothing.

### Option B — Invictus-IR AWS CloudTrail dataset

- Source: https://github.com/invictus-ir/aws_dataset
- Place the JSON files in `Dataset/Raw/CloudTrail/CloudTrail/`

Real CloudTrail from a Stratus Red Team attack simulation. It has no `labels.json`;
labelling falls back to the account-name convention described below.

## Labels

Ground truth is read from `labels.json` in the raw data directory:

```json
{"entities": [
  {"user": "user_042", "label": 1, "scenario": "privilege_escalation",
   "attack_window": ["2024-03-14T09:12:00Z", "2024-03-14T09:41:00Z"]}
]}
```

If no manifest is present, accounts whose name contains `hacker` are treated as
compromised. Both Phase 1 and Phase 5 print which source they used, and raise if it
yields zero anomalies — which is what happens when a `labels.json` from a different
generation is left behind.

Labels must never be derived from a rule over `eventName`. That column also feeds the
`unique_events` feature, so such a label leaks into the model's own inputs.

## Pipeline

| Notebook | Input | Output |
|---|---|---|
| `Phase1_Data_Preparation` | Raw CloudTrail JSON | `df_identity_clean.csv` |
| `Phase2_Feature_Engineering` | `df_identity_clean.csv` | `features.csv` |
| `Phase3_Graph_Construction` | `features.csv` | `graph_data.pt` (HeteroData) |
| `Phase4_HGNN_Model` | `graph_data.pt` | `model_baseline.pt`, k-fold CV metrics |
| `Phase5_Federated_Learning` | `graph_data.pt`, raw JSON | `model_federated.pt`, federated metrics |

Graph schema: `user`, `ip`, and `service` nodes; `user -accesses-> ip` and
`user -calls-> service` edges, plus their reverses so HANConv can update every node
type. Labels live on `user` nodes.

## Reading the results

Both Phase 4 and Phase 5 evaluate on **held-out** nodes only, and print the score a
do-nothing classifier would get. Check that first: with ~15% anomalies, always
answering "normal" already scores ~87% accuracy, so accuracy alone says very little.
Judge runs on precision, recall, and F1 for the anomaly class, and read them next to
the test-set size.

Phase 5 also reports whether the two classes' score ranges overlap. If they don't,
any threshold in the gap scores identically and the run is not testing the model —
it is measuring how separable the dataset already was.

Note that Phase 4 and Phase 5 currently train with different hyperparameters
(hidden 64 / lr 0.01 vs hidden 8 / lr 0.001), so the centralized-vs-federated gap
reflects model capacity as well as federation.

## Layout

```
Dataset/
  generate_synthetic_logs.py   synthetic CloudTrail generator
  Raw/                         raw logs (gitignored)
  Processed/                   intermediate CSVs, graph, models, charts
Notebooks/                     Phase1 - Phase5, run in order
```
