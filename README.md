# BSI-Agent: Interactive LLM Agent for Bloodstream Infection Management

An LLM-based clinical decision support agent for bloodstream infection (BSI) diagnosis and treatment recommendation using MIMIC-IV data.

## Overview

This project develops a text-based Large Language Model (LLM) agent to assist clinical decision-making in bloodstream infection cases. The agent interactively queries patient data (labs, vitals, medications, microbiology results) and progressively refines its diagnostic hypotheses, ultimately suggesting the most likely pathogens and appropriate antibiotic therapy.

## Key Features

- **Pathogen Identification**: Top-K likely pathogens with explanatory reasoning
- **Calibrated Confidence**: Well-calibrated probability estimates (evaluated via Brier score)
- **Data-Driven Reasoning**: All claims grounded in provided patient data
- **Efficient Dialogue**: Concise interactions reaching conclusions in minimal turns
- **Safety Guardrails**: Rule-based checks for allergies and contraindications

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Environment   │◄───►│   LLM Agent     │
│  (MIMIC-IV data)│     │  (Fine-tuned 7B)│
└─────────────────┘     └─────────────────┘
                              │
                              ▼
                        ┌───────────┐
                        │ Guardrails│
                        └───────────┘
```

## Project Structure

```
bsi-agent/
├── src/bsi_agent/
│   ├── data/           # Data loading and preprocessing
│   ├── environment/    # EHR simulation environment
│   ├── agent/          # LLM agent implementation
│   ├── guardrails/     # Safety checks
│   └── evaluation/     # Metrics and evaluation
├── tests/              # Unit tests
├── data/
│   ├── raw/            # Raw MIMIC-IV data (not committed)
│   ├── processed/      # Processed patient cases
│   └── synthetic_dialogues/  # GPT-4 generated training data
├── notebooks/          # Jupyter notebooks for exploration
├── configs/            # Configuration files
└── scripts/            # Utility scripts
```

## Installation

```bash
# Clone the repository
git clone https://github.com/AviYos/bsi-agent.git
cd bsi-agent

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Setup

This project uses MIMIC-IV, which requires credentialed access:

1. Complete CITI training at https://physionet.org/settings/credentialing/
2. Sign the MIMIC-IV Data Use Agreement
3. Download MIMIC-IV from https://physionet.org/content/mimiciv/
4. Place data files in `data/raw/`

See `data/README.md` for detailed instructions.

## Usage

```bash
# Preprocess MIMIC-IV data
python scripts/preprocess_mimic.py

#Full:
python scripts/preprocess_mimic.py --mimic_path /app/data/LLM_project_data/physionet.org/files/mimiciv/3.1 --output_dir data/processed --max_cases 1000 --val_ratio 0.0

# Generate synthetic dialogues
python scripts/generate_dialogues.py --num_dialogues 500 
#Full:
python scripts/generate_dialogues.py --num_dialogues 50 --cases_path /app/code/LLM_project/bsi-agent-ori/data/processed/train_cases.jsonl --api_key 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


# Fine-tune the model
python scripts/train.py --config configs/train_config.yaml
#Full:
python scripts/train.py --config configs/config.yaml

# Run evaluation
python scripts/evaluate.py --model_path outputs/model
#Full: 
python scripts/evaluate.py --model_path outputs/model/checkpoint-5 --test_cases /app/code/LLM_project/bsi-agent/data/processed/test_cases.jsonl --save_chat 1 --adapter_path /app/code/LLM_project/bsi-agent/outputs/model/checkpoint-5 --environment_api_key  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

python scripts/evaluate.py   --base_model mistralai/Mistral-7B-Instruct-v0.2   --adapter_path outputs/model/final   --test_cases /app/code/LLM_project/bsi-agent-ori/data/processed/test_cases.jsonl   --max_turns 5   --environment_model gpt-4o   --save_dialogues outputs/eval_dialogues.jsonl   --environment_api_key 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 --max_cases 10

# If Local:
python scripts/evaluate.py   --base_model mistralai/Mistral-7B-Instruct-v0.2   --adapter_path outputs/model/final   --test_cases /app/code/LLM_project/bsi-agent/data/processed/test_cases.jsonl   --max_turns 5   --environment_model microsoft/Phi-3.5-mini-instruct   --save_dialogues outputs/eval_dialogues.jsonl   --backend local --num_workers 1

python scripts/evaluate.py \
  --agent_type gpt4o \
  --test_cases /app/code/LLM_project/bsi-agent-ori/data/processed/test_cases.jsonl \
  --agent_api_key "sk-YOUR_OPENAI_API_KEY" \
  --environment_api_key "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" \
  --save_dialogues outputs/gpt4o_eval_results.jsonl
```
## Hardware Requirements

- NVIDIA GPU with 48GB VRAM (e.g., L40) for training
- 16GB+ VRAM for inference
- Uses QLoRA for memory-efficient fine-tuning

## Evaluation Metrics

- **Top-K Accuracy**: Correct pathogen in top K predictions
- **Brier Score**: Calibration of confidence estimates
- **Dialogue Efficiency**: Number of turns to reach conclusion
- **Grounding Score**: Percentage of claims supported by data

## License

This project is for research purposes only. MIMIC-IV data usage is subject to the PhysioNet Credentialed Health Data License.

## Acknowledgments

- MIMIC-IV dataset by PhysioNet
- Based on implementation plan for Bar-Ilan University LLM course
