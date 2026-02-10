# BSI-Agent: Training a Language Model to Ask Diagnostic Questions

Train a small language model (Model D) that, given a partial clinical summary of a bloodstream infection (BSI) patient, asks the single most informative diagnostic question.

## Architecture

Four models work together:

| Model | Role | Implementation |
|-------|------|----------------|
| **A** | Summarizer / Answerer | GPT-4o |
| **B** | Question Generator | GPT-4o |
| **C** | Pathogen Classifier | GPT-4o |
| **D** | Fine-tuned Question Generator | Mistral-7B (QLoRA) |

## Pipeline Overview

```
Phase 0: Validate C          Phase 1: Generate Training Data          Phase 2: Train D       Phase 3: Evaluate D
---------------------    ------------------------------------    ------------------    -------------------
Full Summary (100%)      Raw Case + Full Summary                 good [x, q] pairs    D generates q*
        |                        |                                      |              compare q* to q
   Model C classifies      1. Hide ~50% categories                  QLoRA fine-tune    sentence similarity
        |                  2. Model A narrates partial -> x              |
   Top-3 accuracy?         3. Model B asks question -> q           Model D trained
   (target: 70%)           4. Model A answers from full -> a
                           5. d = x + q + a
                           6. Model C classifies x -> rank_x
                           7. Model C classifies d -> rank_d
                           8. Keep [x,q] where rank_d < rank_x
```

---

## End-to-End Example (2 Cases)

### Phase 0 - Validate Model C

Model C receives full clinical summaries (100% patient information) and predicts the top-3 most likely pathogens.

**Result on 100 cases:** Top-1 = 38.4%, **Top-3 = 77.8%** (target: >70%)

### Phase 1 - Dataset Generation

#### Step 1: Create Partial Summaries

Raw case data is randomly partitioned by category. Each category is independently kept or hidden:

| Category | Keep Probability | Hidden in Example? |
|----------|------------------|--------------------|
| Demographics | 100% | No |
| Admission | 80% | No |
| Labs | 40% | No |
| Vitals | 50% | No |
| Medications | 40% | **Yes** |
| Gram Stain | 20% | **Yes** |

Model A then generates a narrative clinical summary from the partial case data, producing `x` (the partial summary).

**Case 2 example** (67M, true pathogen: Beta Streptococcus Group B):
> The patient is a 67-year-old male admitted on March 14, 2146... elevated WBC at 16.1 K/uL... creatinine elevated at 5.4 mg/dL... laboratory findings suggest a suspected bloodstream infection with accompanying renal impairment...

Note: medications and gram stain were hidden, but the summary does **not** mention their absence — it simply omits them. This prevents data leakage that would hint which question to ask.

#### Step 2: Model B Asks a Question

Model B sees only the partial summary `x` and asks one diagnostic question `q`:

> **q:** "What was the result of the Gram stain morphology from the blood culture?"

Model B used clinical reasoning to identify gram stain as the most valuable missing information, without being tipped off by the summary.

#### Step 3: Model A Answers

Model A sees the **full** patient data (including hidden categories) and answers:

> **a:** "The gram stain morphology observed in the initial blood culture showed gram-positive cocci in chains."

#### Step 4: Create Dialogue

The dialogue `d` is assembled as: `d = x + q + a`

#### Step 5-6: Model C Classifies

Model C predicts the pathogen from the partial summary alone, and from the full dialogue:

| Input | Predictions | Rank |
|-------|-------------|------|
| **x** (partial only) | [E. coli, S. aureus, K. pneumoniae] | **99** (not found) |
| **d** (with Q&A) | [S. pyogenes, **S. agalactiae**, E. faecalis] | **2** (correct at position 2) |

Without the question, Model C had no idea it was Group B Strep. After learning the gram stain showed "gram-positive cocci in chains", it correctly narrowed to Streptococcus species.

#### Step 7: Filter Good Questions

A question is "good" if it improved the pathogen ranking: `rank_d < rank_x`

| Case | Pathogen | rank_x | rank_d | Good? | Improvement |
|------|----------|--------|--------|-------|-------------|
| 1 (79F) | E. coli | 1 | 1 | No | 0 (already correct) |
| **2 (67M)** | **Group B Strep** | **99** | **2** | **Yes** | **97** |

**Result: 1 good [x, q] pair** -- the question about gram stain morphology dramatically improved classification.

### Phase 2 - Train Model D

Fine-tune a small model on good [x, q] pairs using QLoRA:
- Input: partial summary (x)
- Target: the good question (q)
- Loss: only on assistant response (user prompt tokens masked with -100)

```
Training config:
  Base model: TinyLlama-1.1B (debug) / Mistral-7B (production)
  LoRA: r=16, alpha=32, targets=[q_proj, k_proj, v_proj, o_proj]
  Loss: causal LM, supervised only on q (label masking on prompt)
```

### Phase 3 - Evaluate Model D

D receives partial summaries from the test set, generates q\*, and we measure similarity to the reference q:

| Metric | Score | Notes |
|--------|-------|-------|
| Sentence Similarity | 0.31 | Cosine similarity (all-MiniLM-L6-v2) |
| BLEU | 0.01 | Lexical overlap |
| ROUGE-L | 0.11 | Longest common subsequence |
| BERTScore F1 | 0.84 | Semantic similarity (roberta-large) |

Low scores are expected with 1 training sample on a 1.1B model. With 1000 cases and Mistral-7B, scores will improve substantially.

---

## How to Run

### Prerequisites

1. **MIMIC-IV data** in `full_data/mimic-iv-3.1/`
2. **OpenAI API key** in `configs/config.yaml`
3. **Dependencies:** `pip install -r requirements.txt`

### Phase 0: Validate Model C

```bash
python scripts/extract_bsi_cases.py --max_cases 100
python scripts/generate_summaries.py
python scripts/test_classifier.py
python scripts/evaluate_classifier.py
```

### Phase 1: Generate Training Data

```bash
python scripts/generate_training_data.py --max_cases 1000
```

Runs all 7 steps. Use `--step N` to resume from step N. Use `--max_cases 2` for debug.

### Phase 2: Train Model D

```bash
# Debug (CPU, small model)
python scripts/train.py --mode xq --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --num_epochs 3 --batch_size 1 --no_quantize --output_dir outputs/model_debug

# Production (GPU, 4-bit quantization)
python scripts/train.py --mode xq --base_model mistralai/Mistral-7B-Instruct-v0.3
```

### Phase 3: Evaluate Model D

```bash
# Debug
python scripts/evaluate_question_generation.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --adapter_path outputs/model_debug/final \
    --input_path data/processed/good_questions_test.jsonl

# Production
python scripts/evaluate_question_generation.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --adapter_path outputs/model/final \
    --load_in_4bit
```

---

## Output Files

| File | Description |
|------|-------------|
| `data/processed/bsi_cases.jsonl` | Raw BSI cases from MIMIC-IV |
| `data/processed/full_summaries.jsonl` | Full clinical summaries (Model A) |
| `data/processed/classification_results.jsonl` | Phase 0: Model C predictions on full summaries |
| `data/processed/validation_report.json` | Phase 0: accuracy report |
| `data/processed/partial_summaries.jsonl` | Partial summaries with hidden categories |
| `data/processed/questions.jsonl` | Questions from Model B |
| `data/processed/answers.jsonl` | Answers from Model A |
| `data/processed/dialogues.jsonl` | Assembled dialogues (d = x + q + a) |
| `data/processed/classifications_x.jsonl` | Model C predictions from partial summary (x) |
| `data/processed/classifications_d.jsonl` | Model C predictions from dialogue (d) |
| `data/processed/good_questions.jsonl` | Filtered good [x, q] pairs |
| `data/processed/good_questions_train.jsonl` | Training split (80%) |
| `data/processed/good_questions_test.jsonl` | Test split (20%) |
| `data/processed/question_eval.json` | Evaluation metrics |
| `outputs/model/final/` | Trained LoRA adapter |

---

## Project Structure

```
bsi-agent/
├── configs/
│   └── config.yaml                          # API keys & model settings
├── scripts/
│   ├── extract_bsi_cases.py                 # Extract BSI cases from MIMIC-IV
│   ├── generate_summaries.py                # Generate full summaries (Model A)
│   ├── test_classifier.py                   # Test Model C on full summaries
│   ├── evaluate_classifier.py               # Evaluate Phase 0 accuracy
│   ├── generate_training_data.py            # Phase 1: 7-step data generation
│   ├── train.py                             # Phase 2: QLoRA fine-tuning
│   └── evaluate_question_generation.py      # Phase 3: evaluate Model D
├── src/bsi_agent/
│   ├── data/
│   │   ├── mimic_loader.py                  # MIMIC-IV data loading
│   │   ├── bsi_cohort.py                    # BSI case extraction
│   │   ├── utils.py                         # load_jsonl, save_jsonl
│   │   └── redaction.py                     # Sanitize pathogen mentions
│   ├── generation/
│   │   ├── summary_generator.py             # Model A: summarize
│   │   ├── question_generator.py            # Model B: ask questions
│   │   ├── answer_generator.py              # Model A: answer questions
│   │   ├── pathogen_classifier.py           # Model C: classify pathogens
│   │   └── partial_summary.py               # Category-level data hiding
│   └── evaluation/
│       ├── pathogen_matching.py             # Pathogen name alias matching
│       └── similarity_metrics.py            # Sentence similarity, BLEU, ROUGE, BERTScore
└── data/processed/                          # All pipeline outputs
```

---

## Cost Estimate (1000 cases)

| Step | API Calls | Est. Cost |
|------|-----------|-----------|
| Extract + Full Summaries | 1000 | ~$10 |
| Partial Summaries (Model A) | 1000 | ~$10 |
| Questions (Model B) | 1000 | ~$5 |
| Answers (Model A) | 1000 | ~$5 |
| Classify from x (Model C) | 1000 | ~$5 |
| Classify from d (Model C) | 1000 | ~$5 |
| **Total** | **6000** | **~$40** |

---

## Hardware Requirements

- **Training (production):** NVIDIA GPU with 16GB+ VRAM (QLoRA with 4-bit quantization)
- **Training (debug):** CPU only (use `--no_quantize` with TinyLlama-1.1B)
- **Inference:** 8GB+ VRAM with 4-bit quantization

## License

This project is for research purposes only. MIMIC-IV data usage is subject to the PhysioNet Credentialed Health Data License.

## Acknowledgments

- MIMIC-IV dataset by PhysioNet
- Bar-Ilan University LLM Course Project
