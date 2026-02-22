# BSI-Agent: Teaching a Small LLM to Ask the Right Diagnostic Question

> **The problem:** A patient has a bloodstream infection. The clinician has incomplete lab results. What single question should they ask to best identify the pathogen?

This project trains a small language model (Model D) to generate that question. Given a partial clinical summary with missing data, Model D learns to ask the one question that most improves pathogen identification -- distilling the diagnostic reasoning of GPT-4o into a model that can run locally.

**Built on:** MIMIC-IV (10,000+ real ICU cases) | **Course:** Bar-Ilan University, LLM Course

---

## The Core Idea

```
 Partial patient summary               "What was the Gram stain
 (labs, vitals -- but some     --->     morphology from the blood
  data is missing)              D        culture?"
```

Model D doesn't guess the pathogen. It asks the question that *helps a classifier guess better*.

---

## How It Works: 4 Models, 1 Pipeline

```
                                    TRAINING PIPELINE
                    ================================================

  Raw Case (100% data)
        |
        v
  [1] Hide ~50% of data ---------> Partial Summary (x)
        |                                  |
        |                                  v
        |                          [2] Model B asks: "What is the Gram stain?" ---> q
        |                                  |
        v                                  v
  [3] Model A answers from full data ----> a = "Gram-positive cocci in chains"
                                           |
                                           v
                                    [4] Dialogue: d = x + q + a
                                           |
                              +------------+------------+
                              |                         |
                              v                         v
                    [5] Model C classifies      [6] Model C classifies
                        from x alone                from d (with Q&A)
                        rank_x = 99                 rank_d = 2
                              |                         |
                              +------------+------------+
                                           |
                                           v
                                    [7] rank_d < rank_x?
                                        YES --> good [x, q] pair
                                           |
                                           v
                                    [8] Train Model D on
                                        good [x, q] pairs (QLoRA)
```

| Model | Role | Implementation |
|-------|------|----------------|
| **A** | Summarizer + Answerer | GPT-4o -- narrates partial data, answers questions from full data |
| **B** | Question Asker | GPT-4o -- given partial summary + hints of what's hidden, asks one question |
| **C** | Pathogen Classifier | GPT-4o -- predicts top pathogens from clinical text |
| **D** | Learned Question Asker | Mistral-7B with QLoRA -- the model we're training |

---

## Example: One Case Through the Pipeline

**Patient:** 67-year-old male, admitted to ICU. True pathogen: **Beta Streptococcus Group B**.

**Step 1 -- Partial Summary (x):** Labs show elevated WBC (16.1), creatinine (5.4). Medications and gram stain are **hidden**.

**Step 2 -- Model B asks (q):**
> *"What was the result of the Gram stain morphology from the blood culture?"*

**Step 3 -- Model A answers (a):**
> *"Gram-positive cocci in chains."*

**Step 5-6 -- Classification impact:**

| Input | Model C's Top Predictions | Rank of True Pathogen |
|-------|--------------------------|----------------------|
| Partial only (x) | E. coli, K. pneumoniae, S. aureus | **99** (not found) |
| With Q&A (d) | S. pyogenes, **S. agalactiae**, E. faecalis | **2** |

One question moved the pathogen from "not even in the top 10" to position 2. This is a **good question** -- it becomes a training example for Model D.

---

## Key Design Decisions

### Data Hiding (Partial Summaries)
Each clinical item is independently kept or hidden at the **item level** (not category level):

| Category | Keep % | Rationale |
|----------|--------|-----------|
| Demographics | 100% | Age/gender are always available |
| Admission | 100% | ICU vs ward context is fundamental |
| Labs | 40% | Force the model to ask for missing labs |
| Vitals | 50% | Partially available in practice |
| Medications | 50% | Treatment context is sometimes known |
| Gram Stain | 0% | Key diagnostic info -- always hide |
| Organism | 0% | The answer -- never reveal |
| Susceptibilities | 0% | Would leak the pathogen identity |

Hidden items generate **hints** passed to Model B, ensuring it only asks about data that exists in the full record.

### Style Variation
Each summary and question is generated with a **randomized writing style** (tone, structure, specificity). This prevents Model D from memorizing surface patterns and improves generalization.

### Label Masking
During training, loss is computed **only on the question tokens** (the assistant response). The partial summary tokens in the prompt are masked with -100 so the model learns *what to ask*, not *how to repeat the input*.

---

## Results

### Phase 0: Model C Baseline (100 cases, full summaries)

| Metric | Score |
|--------|-------|
| Top-1 Accuracy | 38.4% |
| **Top-3 Accuracy** | **77.8%** |
| Target | 70% |

Model C can identify the pathogen in the top 3 for ~78% of cases when given the full summary. This validates it as a useful classifier for the pipeline.

### Phase 1: Question Quality

Questions are filtered by whether they improve classification. The **good question rate** and **average rank improvement** are the key metrics for the dataset quality.

### Phase 3: Model D Evaluation

Model D is evaluated on held-out test cases using 4 metrics:

| Metric | What It Measures |
|--------|-----------------|
| Sentence Similarity | Semantic closeness (all-MiniLM-L6-v2) |
| BLEU | Lexical n-gram overlap |
| ROUGE-L | Longest common subsequence |
| BERTScore F1 | Contextual semantic similarity (roberta-large) |

---

## How to Run

### Prerequisites

1. **MIMIC-IV data** in `full_data/mimic-iv-3.1/`
2. **OpenAI API key** in `configs/config.yaml`
3. `pip install -r requirements.txt`

### Phase 0: Validate Model C

```bash
python scripts/extract_bsi_cases.py --max_cases 100
python scripts/generate_summaries.py
python scripts/test_classifier.py
python scripts/evaluate_classifier.py
```

### Phase 1: Generate Training Data

```bash
python scripts/generate_training_data.py --max_cases 1000           # full run
python scripts/generate_training_data.py --step 3 --max_cases 1000  # resume from step 3
python scripts/generate_training_data.py --max_cases 2              # debug
python scripts/generate_training_data.py --max_cases 1000 --workers 10  # more threads
```

All 7 steps run in sequence. Each step checkpoints to disk, so you can resume with `--step N`. Execution logs go to `logs/`.

### Phase 2: Train Model D

```bash
# Debug (CPU, small model)
python scripts/train.py --mode xq \
    --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --num_epochs 3 --batch_size 1 --no_quantize \
    --output_dir outputs/model_debug

# Production (GPU, Mistral-7B with 4-bit QLoRA)
python scripts/train.py --mode xq \
    --base_model mistralai/Mistral-7B-Instruct-v0.3
```

### Phase 3: Evaluate Model D

```bash
python scripts/evaluate_question_generation.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --adapter_path outputs/model/final \
    --load_in_4bit --split test
```

### Plots for Presentation

```bash
python scripts/create_plots.py
```

Generates 7 figures to `outputs/plots/`: pathogen distribution, Top-K accuracy, per-pathogen breakdown, rank improvement (x vs d), question filtering, quality radar chart, and training loss curve.

---

## Project Structure

```
bsi-agent/
├── configs/config.yaml                 # API keys, model & training config
│
├── scripts/
│   ├── extract_bsi_cases.py            # Extract BSI cases from MIMIC-IV
│   ├── generate_summaries.py           # Generate full summaries (Model A)
│   ├── test_classifier.py              # Test Model C on full summaries
│   ├── evaluate_classifier.py          # Evaluate Phase 0 accuracy
│   ├── generate_training_data.py       # 7-step parallel data generation pipeline
│   ├── train.py                        # QLoRA fine-tuning (xq mode)
│   ├── evaluate_question_generation.py # Evaluate Model D (4 metrics)
│   ├── create_plots.py                 # Generate presentation plots
│   └── demo_one_case.py               # Demo: one case end-to-end
│
├── src/bsi_agent/
│   ├── data/                           # MIMIC-IV loading, case extraction, redaction
│   ├── generation/                     # Models A, B, C + partial summary + style variation
│   └── evaluation/                     # Pathogen matching + similarity metrics
│
├── data/processed/                     # All pipeline outputs (JSONL)
├── outputs/                            # Trained adapters + plots
└── logs/                               # Execution logs
```

<details>
<summary>Output files reference</summary>

| File | Description |
|------|-------------|
| `data/processed/bsi_cases.jsonl` | Raw BSI cases from MIMIC-IV |
| `data/processed/full_summaries.jsonl` | Full clinical summaries |
| `data/processed/classification_results.jsonl` | Phase 0 predictions |
| `data/processed/partial_summaries.jsonl` | Partial summaries + hints |
| `data/processed/questions.jsonl` | Model B questions |
| `data/processed/answers.jsonl` | Model A answers |
| `data/processed/dialogues.jsonl` | Assembled dialogues (d = x + q + a) |
| `data/processed/classifications_x.jsonl` | Classify from partial (x) |
| `data/processed/classifications_d.jsonl` | Classify from dialogue (d) |
| `data/processed/good_questions.jsonl` | Filtered good [x, q] pairs |
| `data/processed/question_eval.json` | Model D evaluation results |
| `outputs/model/final/` | Trained LoRA adapter |
| `outputs/plots/` | Presentation figures |

</details>

---

## Cost & Hardware

| | Detail |
|-|--------|
| **API cost** (1000 cases) | ~$40 (6 GPT-4o calls per case) |
| **Training (production)** | NVIDIA GPU, 16GB+ VRAM (QLoRA 4-bit) |
| **Training (debug)** | CPU only (`--no_quantize` with TinyLlama-1.1B) |
| **Inference** | 8GB+ VRAM with 4-bit quantization |

---

## License

Research purposes only. MIMIC-IV data usage is subject to the PhysioNet Credentialed Health Data License.

## Acknowledgments

- MIMIC-IV dataset (PhysioNet)
- Bar-Ilan University LLM Course
