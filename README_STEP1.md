# Step 1: Validate Model C's Pathogen Classification

## Overview

This step validates that GPT-4o (Model C) can accurately classify bloodstream infection pathogens from clinical summaries. This is a prerequisite before building the full dataset generation pipeline.

**Target:** >70% Top-3 Accuracy
**Result:** 86.9% ✅

---

## Pipeline

```
MIMIC-IV Data
     │
     ▼
┌─────────────────────────────┐
│ Step 1.1: Extract BSI Cases │  (No API needed)
│ - Positive blood cultures   │
│ - Labs, vitals, medications │
│ - Ground truth pathogen     │
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│ Step 1.2: Generate Summaries│  (GPT-4o / Model A)
│ - 100% patient information  │
│ - Clinical narrative format │
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│ Step 1.3: Test Classifier   │  (GPT-4o / Model C)
│ - Predict top 3 pathogens   │
│ - From full summaries       │
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│ Step 1.4: Evaluate Accuracy │  (No API needed)
│ - Compare to ground truth   │
│ - Report top-3 accuracy     │
└─────────────────────────────┘
```

---

## Prerequisites

### 1. MIMIC-IV Data
Place MIMIC-IV v3.1 data in:
```
bsi-agent/full_data/mimic-iv-3.1/
├── hosp/
│   ├── microbiologyevents.csv.gz
│   ├── patients.csv.gz
│   ├── admissions.csv.gz
│   ├── labevents.csv.gz
│   └── prescriptions.csv.gz
└── icu/
    └── chartevents.csv.gz (optional)
```

### 2. OpenAI API Key
Add your API key to `configs/config.yaml`:
```yaml
dialogue_generation:
  api_key: sk-your-api-key-here
  model: gpt-4o
```

### 3. Python Dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

### Extract BSI Cases
Extracts positive blood cultures with clinical context from MIMIC-IV.

```bash
python scripts/extract_bsi_cases.py --max_cases 100
```

**Options:**
- `--max_cases`: Number of cases to extract (default: 100)
- `--mimic_path`: Path to MIMIC-IV data (default: full_data/mimic-iv-3.1)
- `--output_path`: Output file (default: data/processed/bsi_cases.jsonl)

**Output:** `data/processed/bsi_cases.jsonl`

---

### Generate Summaries
Generates clinical narrative summaries using GPT-4o (Model A).

```bash
python scripts/generate_summaries.py
```

**Options:**
- `--cases_path`: Input cases (default: data/processed/bsi_cases.jsonl)
- `--output_path`: Output file (default: data/processed/full_summaries.jsonl)
- `--max_cases`: Limit cases to process (optional)

**Output:** `data/processed/full_summaries.jsonl`

**Cost:** ~$0.01-0.02 per case

---

### Test Classifier
Tests GPT-4o's ability to classify pathogens (Model C).

```bash
python scripts/test_classifier.py
```

**Options:**
- `--summaries_path`: Input summaries (default: data/processed/full_summaries.jsonl)
- `--output_path`: Output file (default: data/processed/classification_results.jsonl)

**Output:** `data/processed/classification_results.jsonl`

---

### Evaluate Results
Computes top-3 accuracy with proper pathogen name matching.

```bash
python scripts/evaluate_classifier.py
```

**Options:**
- `--results_path`: Input results (default: data/processed/classification_results.jsonl)
- `--output_path`: Output report (default: data/processed/validation_report.json)

**Output:** `data/processed/validation_report.json`

---

## Run All Steps

```bash
python scripts/extract_bsi_cases.py --max_cases 100
python scripts/generate_summaries.py
python scripts/test_classifier.py
python scripts/evaluate_classifier.py
```

---

## Output Files

| File | Description |
|------|-------------|
| `data/processed/bsi_cases.jsonl` | Extracted BSI cases with structured data |
| `data/processed/full_summaries.jsonl` | Generated clinical summaries |
| `data/processed/classification_results.jsonl` | Model C's pathogen predictions |
| `data/processed/validation_report.json` | Final accuracy report |

---

## Results (100 cases)

```
TOP-3 ACCURACY: 86/99 = 86.9%
[PASS] Target accuracy (70%) achieved!
```

### By Pathogen:
| Pathogen | Accuracy |
|----------|----------|
| Escherichia coli | 100% |
| Staphylococcus aureus | 90% |
| Klebsiella pneumoniae | 100% |
| Enterococcus spp. | 100% |
| CoNS | 100% |
| Rare pathogens | Lower |

---

## File Structure

```
bsi-agent/
├── scripts/
│   ├── extract_bsi_cases.py
│   ├── generate_summaries.py
│   ├── test_classifier.py
│   └── evaluate_classifier.py
├── src/bsi_agent/
│   ├── data/
│   │   ├── mimic_loader.py        # MIMIC-IV data loading
│   │   └── bsi_cohort.py          # BSI case extraction
│   └── generation/
│       ├── summary_generator.py   # Model A
│       └── pathogen_classifier.py # Model C
├── configs/
│   └── config.yaml                # API key & settings
└── data/processed/
    ├── bsi_cases.jsonl
    ├── full_summaries.jsonl
    ├── classification_results.jsonl
    └── validation_report.json
```

---

## Next Steps

With Model C validated (86.9% accuracy), proceed to:

**Step 2: Dataset Generation**
- Generate partial summaries (50% info hidden)
- Model B asks diagnostic questions
- Model A answers questions
- Create (x, q, d) tuples

**Step 3: Quality Labeling**
- Model C classifies from partial summary (x) → accuracy_x
- Model C classifies from dialogue (d) → accuracy_d
- Keep [x, q] pairs where accuracy_d > accuracy_x

**Step 4: Fine-tune Model D**
- Train Mistral-7B on good [x, q] pairs
