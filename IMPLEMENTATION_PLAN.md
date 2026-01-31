# BSI-Agent Implementation Plan

## Project Goal
Train a language model (D) that asks the right diagnostic questions. Given a partial medical summary, the model generates a single question to obtain the most important missing information for pathogen identification.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: Dataset Generation                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  MIMIC-IV Data ──► Model A (GPT-4o) ──► Full Summary (100%)         │
│                         │                                            │
│                         ▼                                            │
│                    Partial Summary (50%) ──► Model B (GPT-4o)       │
│                         │                        │                   │
│                         │                        ▼                   │
│                         │                    Question (q)            │
│                         │                        │                   │
│                         ▼                        ▼                   │
│                    Model A answers with focused response             │
│                         │                                            │
│                         ▼                                            │
│              Output: (x=summary, q=question, d=full_dialogue)       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: Quality Labeling                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Model C (GPT-4o) classifies pathogen from:                         │
│    • x only (partial summary) → accuracy_x                          │
│    • d (full dialogue)        → accuracy_d                          │
│                                                                      │
│  IF accuracy_d > accuracy_x:                                        │
│    KEEP pair [x, q] as "good question"                              │
│  ELSE:                                                              │
│    DISCARD (question didn't help)                                   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 3: Fine-tuning Model D                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Model D (Mistral-7B) fine-tuned on [x, q] pairs                    │
│  Input: partial medical summary                                      │
│  Output: diagnostic question                                         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 4: Evaluation                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Test set: D generates q* from x                                     │
│  Metric: Sentence similarity between q* and ground-truth q          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Steps

### Phase 0: Data Preparation & Validation (PREREQUISITE)
**Goal**: Prepare MIMIC-IV data and validate that Model C can classify pathogens accurately.

#### Step 0.1: Extract BSI Cases from MIMIC-IV
- **Input**: `full_data/mimic-iv-3.1/hosp/microbiologyevents.csv.gz`
- **Task**: Filter for blood culture specimens with positive pathogen results
- **Output**: List of `(subject_id, hadm_id, pathogen)` tuples
- **Key tables**:
  - `microbiologyevents.csv.gz` - pathogen identification (org_name field)
  - `admissions.csv.gz` - admission details
  - `patients.csv.gz` - demographics
  - `labevents.csv.gz` - lab results
  - `diagnoses_icd.csv.gz` - diagnoses

#### Step 0.2: Build Patient Case Objects
- **Task**: For each BSI case, aggregate all relevant clinical data:
  - Demographics (age, gender)
  - Admission info (admit time, discharge time, admission type)
  - Lab results (prior to culture date)
  - Vital signs (from ICU chartevents if applicable)
  - Medications/prescriptions
  - Diagnoses
  - Microbiology results (ground truth pathogen)
- **Output**: `processed/bsi_cases.jsonl` - complete patient cases

#### Step 0.3: Validate Model C's Classification Ability
- **Task**: Test if GPT-4o can correctly identify pathogens from full summaries
- **Method**:
  1. Generate 100% complete summaries using Model A
  2. Ask Model C to rank likely pathogens
  3. Compare against ground truth
- **Success criteria**: >70% top-3 accuracy on pathogen identification
- **Output**: Baseline accuracy metrics

---

### Phase 1: Dataset Generation
**Goal**: Create synthetic dialogues where good questions improve pathogen prediction.

#### Step 1.1: Generate Full Medical Summaries (Model A)
- **Input**: BSI case objects
- **Task**: Prompt GPT-4o to create comprehensive clinical summaries
- **Prompt template**:
```
You are a clinical documentation specialist. Given the following patient data,
write a comprehensive medical summary suitable for clinical decision-making.

Patient Data:
{structured_patient_data}

Write a detailed clinical summary including:
- Patient demographics and chief complaint
- Relevant medical history
- Current presentation and vital signs
- Laboratory findings
- Current medications
- Clinical assessment
```
- **Output**: `summaries/full_summaries.jsonl`

#### Step 1.2: Create Partial Summaries (50%)
- **Task**: Systematically hide 50% of information from summaries
- **Strategy options**:
  - Random sentence removal
  - Category-based hiding (hide labs OR vitals OR meds)
  - Importance-weighted hiding (hide more critical info)
- **Output**: `summaries/partial_summaries.jsonl`

#### Step 1.3: Generate Questions (Model B)
- **Input**: Partial summaries
- **Task**: Prompt GPT-4o to ask the most important diagnostic question
- **Prompt template**:
```
You are an infectious disease specialist reviewing a partial patient summary.
Your task is to ask ONE question about the most important missing information
that would help identify the causative pathogen.

Partial Summary:
{partial_summary}

Ask a single, specific clinical question:
```
- **Output**: Questions paired with partial summaries

#### Step 1.4: Generate Answers (Model A)
- **Input**: Full summary + question
- **Task**: Answer the question using information from the full summary
- **Prompt template**:
```
Based on the complete patient record, answer the following clinical question
with specific, relevant information.

Question: {question}

Full Patient Record:
{full_summary}

Provide a focused, informative answer:
```
- **Output**: Complete dialogues `(x, q, answer, d)`

---

### Phase 2: Quality Labeling
**Goal**: Filter for questions that actually improve pathogen prediction.

#### Step 2.1: Classify from Partial Summary (Model C on x)
- **Input**: Partial summary only
- **Task**: Predict pathogen ranking
- **Prompt template**:
```
You are an infectious disease specialist. Based on the following clinical summary,
rank the top 5 most likely pathogens causing this bloodstream infection.

Clinical Summary:
{partial_summary}

Provide your ranking as:
1. [Pathogen] - [confidence %]
2. [Pathogen] - [confidence %]
...
```
- **Output**: Pathogen predictions with confidence scores

#### Step 2.2: Classify from Full Dialogue (Model C on d)
- **Input**: Full dialogue (partial summary + question + answer)
- **Task**: Predict pathogen ranking
- **Same prompt structure as 2.1**

#### Step 2.3: Compare and Filter
- **Task**: Keep only pairs where dialogue improves prediction
- **Logic**:
```python
def is_good_question(pred_x, pred_d, true_pathogen):
    rank_x = get_rank(pred_x, true_pathogen)
    rank_d = get_rank(pred_d, true_pathogen)
    return rank_d < rank_x  # Lower rank = better (1 is best)
```
- **Output**: `training_data/good_questions.jsonl` with [x, q] pairs

---

### Phase 3: Fine-tune Model D
**Goal**: Train Mistral-7B to generate good diagnostic questions.

#### Step 3.1: Prepare Training Data
- **Format**: Convert to instruction-tuning format
```json
{
  "instruction": "Given this partial medical summary, ask the most important diagnostic question.",
  "input": "{partial_summary}",
  "output": "{good_question}"
}
```

#### Step 3.2: Fine-tune with QLoRA
- **Base model**: `mistralai/Mistral-7B-Instruct-v0.2`
- **Method**: QLoRA (4-bit quantization + LoRA adapters)
- **Hyperparameters**:
  - Learning rate: 2e-4
  - Batch size: 4 (with gradient accumulation)
  - LoRA rank: 64
  - LoRA alpha: 128
  - Epochs: 3

#### Step 3.3: Save Model
- **Output**: `outputs/model/question_generator/`

---

### Phase 4: Evaluation
**Goal**: Measure how well Model D generates relevant questions.

#### Step 4.1: Generate Questions on Test Set
- **Input**: Held-out test partial summaries
- **Task**: Generate questions using fine-tuned Model D

#### Step 4.2: Compute Similarity Metrics
- **Metrics**:
  - Sentence similarity (using sentence-transformers)
  - BLEU score
  - ROUGE score
  - BERTScore

#### Step 4.3: End-to-End Evaluation
- **Task**: Verify generated questions actually improve pathogen prediction
- **Method**: Same pipeline as Phase 2, using generated questions

---

## File Structure

```
bsi-agent/
├── src/bsi_agent/
│   ├── data/
│   │   ├── mimic_loader.py        # Load and filter MIMIC-IV
│   │   ├── case_builder.py        # Build patient case objects
│   │   └── data_utils.py          # Utilities
│   ├── generation/
│   │   ├── summary_generator.py   # Model A - summaries
│   │   ├── question_generator.py  # Model B - questions
│   │   └── answer_generator.py    # Model A - answers
│   ├── labeling/
│   │   ├── pathogen_classifier.py # Model C - classification
│   │   └── quality_filter.py      # Filter good questions
│   ├── training/
│   │   ├── prepare_data.py        # Format for training
│   │   └── train_model_d.py       # Fine-tune Mistral
│   └── evaluation/
│       ├── similarity_metrics.py  # Sentence similarity
│       └── end_to_end_eval.py     # Full pipeline eval
├── scripts/
│   ├── 01_extract_bsi_cases.py
│   ├── 02_validate_classifier.py
│   ├── 03_generate_dialogues.py
│   ├── 04_label_and_filter.py
│   ├── 05_train_model_d.py
│   └── 06_evaluate.py
├── configs/
│   ├── data_config.yaml
│   ├── generation_config.yaml
│   └── training_config.yaml
└── data/
    ├── processed/
    │   ├── bsi_cases.jsonl
    │   └── train_test_split/
    ├── summaries/
    │   ├── full_summaries.jsonl
    │   └── partial_summaries.jsonl
    ├── dialogues/
    │   └── raw_dialogues.jsonl
    └── training/
        └── good_questions.jsonl
```

---

## Execution Order

```
[START]
    │
    ▼
┌─────────────────────────────────────┐
│ Phase 0: Data Prep & Validation     │
│ ┌─────────────────────────────────┐ │
│ │ 0.1 Extract BSI cases from MIMIC│ │
│ └─────────────────────────────────┘ │
│              │                      │
│              ▼                      │
│ ┌─────────────────────────────────┐ │
│ │ 0.2 Build patient case objects  │ │
│ └─────────────────────────────────┘ │
│              │                      │
│              ▼                      │
│ ┌─────────────────────────────────┐ │
│ │ 0.3 Validate Model C accuracy   │ │◄── CHECKPOINT: >70% top-3 accuracy?
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Phase 1: Dataset Generation         │
│ ┌─────────────────────────────────┐ │
│ │ 1.1 Generate full summaries (A) │ │
│ └─────────────────────────────────┘ │
│              │                      │
│              ▼                      │
│ ┌─────────────────────────────────┐ │
│ │ 1.2 Create partial summaries    │ │
│ └─────────────────────────────────┘ │
│              │                      │
│              ▼                      │
│ ┌─────────────────────────────────┐ │
│ │ 1.3 Generate questions (B)      │ │
│ └─────────────────────────────────┘ │
│              │                      │
│              ▼                      │
│ ┌─────────────────────────────────┐ │
│ │ 1.4 Generate answers (A)        │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Phase 2: Quality Labeling           │
│ ┌─────────────────────────────────┐ │
│ │ 2.1 Classify from x (Model C)   │ │
│ └─────────────────────────────────┘ │
│              │                      │
│              ▼                      │
│ ┌─────────────────────────────────┐ │
│ │ 2.2 Classify from d (Model C)   │ │
│ └─────────────────────────────────┘ │
│              │                      │
│              ▼                      │
│ ┌─────────────────────────────────┐ │
│ │ 2.3 Filter good questions       │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Phase 3: Fine-tune Model D          │
│ ┌─────────────────────────────────┐ │
│ │ 3.1 Prepare training data       │ │
│ └─────────────────────────────────┘ │
│              │                      │
│              ▼                      │
│ ┌─────────────────────────────────┐ │
│ │ 3.2 Fine-tune Mistral-7B        │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Phase 4: Evaluation                 │
│ ┌─────────────────────────────────┐ │
│ │ 4.1 Generate q* on test set     │ │
│ └─────────────────────────────────┘ │
│              │                      │
│              ▼                      │
│ ┌─────────────────────────────────┐ │
│ │ 4.2 Compute similarity metrics  │ │
│ └─────────────────────────────────┘ │
│              │                      │
│              ▼                      │
│ ┌─────────────────────────────────┐ │
│ │ 4.3 End-to-end evaluation       │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
    │
    ▼
[END]
```

---

## Key MIMIC-IV Tables Usage

| Table | Purpose | Key Fields |
|-------|---------|------------|
| `microbiologyevents` | Ground truth pathogens | `subject_id`, `hadm_id`, `org_name`, `spec_type_desc` |
| `admissions` | Admission context | `subject_id`, `hadm_id`, `admittime`, `admission_type` |
| `patients` | Demographics | `subject_id`, `gender`, `anchor_age` |
| `labevents` | Lab results | `subject_id`, `hadm_id`, `itemid`, `valuenum`, `valueuom` |
| `d_labitems` | Lab item definitions | `itemid`, `label` |
| `diagnoses_icd` | Diagnoses | `subject_id`, `hadm_id`, `icd_code` |
| `prescriptions` | Medications | `subject_id`, `hadm_id`, `drug`, `dose_val_rx` |
| `chartevents` (ICU) | Vitals | `subject_id`, `hadm_id`, `itemid`, `valuenum` |

---

## Next Steps (Start Here)

1. **Step 0.1**: Run `scripts/01_extract_bsi_cases.py` to identify positive blood cultures
2. Review extracted cases and pathogen distribution
3. Proceed to Step 0.2 once cases are validated

Ready to begin implementation?
