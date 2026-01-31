# Step 1: Validate Model C's Pathogen Classification Ability

## Objective
Verify that Model C (GPT-4o) can accurately classify pathogens from **full (100%)** medical summaries before building the entire pipeline.

**Success Criteria**: >70% top-3 accuracy on pathogen identification

---

## Pipeline Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  MIMIC-IV Data  │───►│  Extract BSI    │───►│  Generate Full  │───►│  Model C        │
│                 │    │  Cases          │    │  Summary (A)    │    │  Classifies     │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                      │                      │
                              ▼                      ▼                      ▼
                       Ground Truth            100% Patient            Predicted
                       Pathogen                Information             Pathogen Ranking
                                                                            │
                                                                            ▼
                                                                    ┌─────────────────┐
                                                                    │  Compare &      │
                                                                    │  Report Metrics │
                                                                    └─────────────────┘
```

---

## Detailed Steps

### Step 1.1: Extract BSI Cases from MIMIC-IV
**Script**: `scripts/step1_extract_bsi_cases.py`

**Input**:
- `full_data/mimic-iv-3.1/hosp/microbiologyevents.csv.gz`
- `full_data/mimic-iv-3.1/hosp/patients.csv.gz`
- `full_data/mimic-iv-3.1/hosp/admissions.csv.gz`
- `full_data/mimic-iv-3.1/hosp/labevents.csv.gz`
- `full_data/mimic-iv-3.1/hosp/prescriptions.csv.gz`
- `full_data/mimic-iv-3.1/icu/chartevents.csv.gz` (optional, for vitals)

**Process**:
1. Load microbiologyevents
2. Filter for positive blood cultures (spec_type_desc = "BLOOD CULTURE", org_name is not null)
3. Exclude common contaminants (CoNS, Corynebacterium, etc.)
4. For each positive culture, gather:
   - Patient demographics (age, gender)
   - Admission info (type, location, dates)
   - Lab results (48h window around culture)
   - Vital signs if available
   - Medications/antibiotics
   - Ground truth: `org_name` (the pathogen)

**Output**: `data/processed/bsi_cases.jsonl`
- Start with ~100-200 cases for validation
- Each case has structured data + ground truth pathogen

**Existing Code**: `src/bsi_agent/data/bsi_cohort.py` (BSICohortExtractor) - can be used

---

### Step 1.2: Generate Full Medical Summaries (Model A)
**Script**: `scripts/step2_generate_summaries.py`

**Input**: `data/processed/bsi_cases.jsonl`

**Process**:
1. For each BSI case, create a structured prompt with ALL patient data
2. Call GPT-4o (Model A) to generate a clinical narrative summary
3. Summary should include 100% of available information:
   - Demographics and chief complaint
   - Admission context
   - All lab values with timestamps
   - All vital signs
   - All medications
   - Any other relevant clinical context

**Prompt Template**:
```
You are a clinical documentation specialist. Given the following structured
patient data, write a comprehensive medical summary for a patient with a
suspected bloodstream infection.

Include ALL the following information in narrative form:
- Patient demographics
- Admission details and reason
- Complete laboratory findings with values and units
- Vital signs and trends
- Current medications, especially antibiotics
- Any relevant clinical observations

Patient Data:
{structured_case_data}

Write a detailed clinical summary:
```

**Output**: `data/processed/full_summaries.jsonl`
- Each record: `{case_id, ground_truth_pathogen, full_summary}`

**Cost Estimate**: ~100 cases × $0.01/case = ~$1-2 for GPT-4o

---

### Step 1.3: Test Model C's Classification
**Script**: `scripts/step3_test_classifier.py`

**Input**: `data/processed/full_summaries.jsonl`

**Process**:
1. For each summary, prompt Model C (GPT-4o) to predict pathogens
2. Request ranked list of top-5 most likely pathogens with confidence scores

**Prompt Template**:
```
You are an infectious disease specialist. Based on the following clinical
summary of a patient with a bloodstream infection, predict the most likely
causative pathogen(s).

Clinical Summary:
{full_summary}

Provide your prediction as a ranked list of the top 3 most likely pathogens:
1. [Pathogen name]
2. [Pathogen name]
3. [Pathogen name]

Use standard microbiological nomenclature (e.g., "Staphylococcus aureus", "Escherichia coli").
```

**Output**: `data/processed/classification_results.jsonl`
- Each record: `{case_id, ground_truth, predicted_ranking, raw_response}`

---

### Step 1.4: Evaluate and Report
**Script**: `scripts/step4_evaluate_classifier.py`

**Input**: `data/processed/classification_results.jsonl`

**Metric**:
- **Top-3 Accuracy**: Ground truth pathogen appears in top 3 predictions (TARGET: >70%)

**Output**:
- Console report with top-3 accuracy
- `data/processed/validation_report.json`

---

## File Structure After Step 1

```
bsi-agent/
├── scripts/
│   ├── step1_extract_bsi_cases.py      # NEW
│   ├── step2_generate_summaries.py     # NEW
│   ├── step3_test_classifier.py        # NEW
│   └── step4_evaluate_classifier.py    # NEW
├── src/bsi_agent/
│   ├── data/
│   │   ├── mimic_loader.py             # EXISTS
│   │   └── bsi_cohort.py               # EXISTS
│   └── generation/
│       ├── __init__.py                 # NEW
│       ├── summary_generator.py        # NEW
│       └── pathogen_classifier.py      # NEW
├── data/
│   └── processed/
│       ├── bsi_cases.jsonl             # OUTPUT Step 1.1
│       ├── full_summaries.jsonl        # OUTPUT Step 1.2
│       ├── classification_results.jsonl # OUTPUT Step 1.3
│       ├── validation_report.json      # OUTPUT Step 1.4
│       └── validation_report.md        # OUTPUT Step 1.4
└── configs/
    └── validation_config.yaml          # NEW - API keys, parameters
```

---

## Configuration Needed

**configs/validation_config.yaml**:
```yaml
# Data paths
mimic_path: "full_data/mimic-iv-3.1"
output_dir: "data/processed"

# Extraction settings
max_cases: 100  # Start small for validation
hours_before_culture: 48
hours_after_culture: 24
exclude_contaminants: true

# Model settings
model_a: "gpt-4o"  # Summary generator
model_c: "gpt-4o"  # Pathogen classifier
temperature: 0.3   # Lower for more consistent outputs

# API
openai_api_key: "${OPENAI_API_KEY}"  # From .env file
```

---

## Execution Order

```bash
# 1. Set up environment
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# 2. Extract BSI cases (no API needed)
python scripts/step1_extract_bsi_cases.py --max_cases 100

# 3. Generate summaries (uses GPT-4o)
python scripts/step2_generate_summaries.py

# 4. Test classifier (uses GPT-4o)
python scripts/step3_test_classifier.py

# 5. Evaluate results
python scripts/step4_evaluate_classifier.py
```

---

## Decision Points

After Step 1.4, evaluate results:

| Result | Action |
|--------|--------|
| Top-3 Accuracy > 70% | ✅ Proceed to Step 2 (Dataset Generation) |
| Top-3 Accuracy 50-70% | ⚠️ Improve prompts, try different summary formats |
| Top-3 Accuracy < 50% | ❌ Reconsider approach, may need different model or features |

---

## Ready to Implement?

When you approve this plan, we will implement in order:
1. Step 1.1 - Extract BSI cases
2. Step 1.2 - Generate summaries
3. Step 1.3 - Test classifier
4. Step 1.4 - Evaluate

Each step will be implemented and tested before moving to the next.
