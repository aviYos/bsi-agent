# Step 2: Generate Training Data

## Overview

Generate training data for Model D by creating dialogues and filtering for "good questions" that improve pathogen prediction.

**Input:** 100 full summaries from Step 1
**Output:** ~30-50 good [x, q] pairs for training

---

## Pipeline

```
Full Summary (100%)
       │
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 2.1 Create Partial Summaries                                         │
│     Hide ~50% of info by category (labs, meds, vitals, gram stain)   │
└──────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 2.2 Generate Questions (Model B - GPT-4o)                            │
│     Model B sees partial → asks ONE diagnostic question              │
└──────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 2.3 Generate Answers (Model A - GPT-4o)                              │
│     Model A answers using full patient data                          │
└──────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 2.4 Create Dialogues                                                 │
│     Combine: d = x + q + answer                                      │
└──────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 2.5 Classify from Partial (Model C - GPT-4o)                         │
│     Predict pathogen from x only → rank_x                            │
└──────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 2.6 Classify from Dialogue (Model C - GPT-4o)                        │
│     Predict pathogen from d → rank_d                                 │
└──────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 2.7 Filter Good Questions                                            │
│     Keep [x, q] where rank_d < rank_x (question helped)              │
│     Output: good_questions.jsonl  ← TRAINING DATA                    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## How to Run

### Run All Steps

```bash
python scripts/generate_training_data.py
```

### Resume from a Specific Step

```bash
# Start from step 3 (skip partial summaries and questions)
python scripts/generate_training_data.py --step 3
```

---

## Project Structure

```
src/bsi_agent/
├── data/
│   ├── utils.py                   # load_jsonl, save_jsonl, load_config
│   ├── mimic_loader.py            # MIMIC-IV data loading
│   └── bsi_cohort.py              # BSI case extraction
├── generation/
│   ├── summary_generator.py       # Model A - full summaries
│   ├── pathogen_classifier.py     # Model C - classify pathogens
│   ├── question_generator.py      # Model B - ask questions
│   ├── answer_generator.py        # Model A - answer questions
│   └── partial_summary.py         # Create partial summaries
└── evaluation/
    └── pathogen_matching.py       # Pathogen name matching

scripts/
├── extract_bsi_cases.py           # Step 1.1
├── generate_summaries.py          # Step 1.2
├── test_classifier.py             # Step 1.3
├── evaluate_classifier.py         # Step 1.4
└── generate_training_data.py      # Step 2 (all sub-steps)
```

---

## Output Files

| File | Description |
|------|-------------|
| `partial_summaries.jsonl` | Summaries with 50% hidden |
| `questions.jsonl` | Questions from Model B |
| `answers.jsonl` | Answers from Model A |
| `dialogues.jsonl` | Combined (x, q, answer, d) |
| `classifications_x.jsonl` | Predictions from partial |
| `classifications_d.jsonl` | Predictions from dialogue |
| `good_questions.jsonl` | **FINAL: Training data [x, q]** |

---

## What Makes a "Good" Question?

A question is **good** if it improves pathogen prediction:

```
is_good = (rank_d < rank_x)
```

| rank_x | rank_d | Good? | Explanation |
|--------|--------|-------|-------------|
| 3 | 1 | ✅ YES | Improved from 3rd to 1st |
| 99 | 2 | ✅ YES | Wasn't in top 3, now is |
| 1 | 1 | ❌ NO | No improvement needed |
| 2 | 3 | ❌ NO | Got worse |

---

## Cost Estimate

| Step | API Calls | Cost |
|------|-----------|------|
| 2.2 Questions | 100 | ~$1 |
| 2.3 Answers | 100 | ~$1 |
| 2.5 Classify x | 100 | ~$0.50 |
| 2.6 Classify d | 100 | ~$0.50 |
| **Total** | 400 | **~$3** |

---

## Expected Results

- **Input:** 100 cases
- **Good questions:** ~30-50 (30-50%)
- **Training pairs:** ~30-50 [x, q]

---

## Next Step

After Step 2, proceed to **Step 3: Train Model D**
- Fine-tune Mistral-7B on good [x, q] pairs
- Input: partial summary → Output: diagnostic question
