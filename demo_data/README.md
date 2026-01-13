# Data Directory

This directory contains data for the BSI-Agent project.

## Directory Structure

```
data/
├── raw/                    # Raw MIMIC-IV data (DO NOT COMMIT)
├── processed/              # Processed patient cases
└── synthetic_dialogues/    # GPT-4 generated training dialogues
```

## MIMIC-IV Data Setup

MIMIC-IV is a restricted dataset requiring credentialed access. Follow these steps:

### Step 1: Get PhysioNet Credentialing

1. Create an account at https://physionet.org/
2. Go to https://physionet.org/settings/credentialing/
3. Complete the required training:
   - CITI "Data or Specimens Only Research" course
   - This typically takes 2-4 hours
4. Submit your credentialing application
5. Wait for approval (usually 1-2 weeks)

### Step 2: Sign Data Use Agreement

1. Once credentialed, go to https://physionet.org/content/mimiciv/
2. Click "Request Access"
3. Sign the Data Use Agreement
4. Wait for approval

### Step 3: Download Required Tables

For this project, you need the following MIMIC-IV tables:

**From `hosp` module:**
- `patients.csv.gz` - Patient demographics
- `admissions.csv.gz` - Hospital admissions
- `labevents.csv.gz` - Laboratory results
- `microbiologyevents.csv.gz` - Microbiology cultures (CRITICAL)
- `prescriptions.csv.gz` - Medications
- `diagnoses_icd.csv.gz` - Diagnoses

**From `icu` module:**
- `chartevents.csv.gz` - Vital signs (large file)
- `icustays.csv.gz` - ICU stays

**Reference tables:**
- `d_labitems.csv.gz` - Lab item definitions
- `d_items.csv.gz` - Chart item definitions

### Step 4: Place Files

Download and place files in `data/raw/`:

```
data/raw/
├── hosp/
│   ├── patients.csv.gz
│   ├── admissions.csv.gz
│   ├── labevents.csv.gz
│   ├── microbiologyevents.csv.gz
│   ├── prescriptions.csv.gz
│   └── diagnoses_icd.csv.gz
├── icu/
│   ├── chartevents.csv.gz
│   └── icustays.csv.gz
└── hosp/
    ├── d_labitems.csv.gz
    └── d_items.csv.gz
```

### Alternative: PostgreSQL Setup

For better query performance, load MIMIC-IV into PostgreSQL:

```bash
# Create database
createdb mimiciv

# Load data using official scripts
# https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/buildmimic/postgres
```

Then update `configs/database.yaml` with your connection details.

## Important Notes

- **NEVER commit raw MIMIC data to git** - it's in .gitignore
- **NEVER share MIMIC data** - violates the Data Use Agreement
- Data must be stored securely and deleted after project completion
- All analysis must comply with HIPAA and PhysioNet requirements

## Processed Data

After running preprocessing scripts, processed data will be in `data/processed/`:

```
data/processed/
├── bsi_cases.parquet       # Extracted BSI patient cases
├── train_cases.parquet     # Training set
├── val_cases.parquet       # Validation set
└── test_cases.parquet      # Test set
```

## Synthetic Dialogues

Generated training dialogues will be stored in `data/synthetic_dialogues/`:

```
data/synthetic_dialogues/
├── dialogues_batch_001.jsonl
├── dialogues_batch_002.jsonl
└── ...
```
