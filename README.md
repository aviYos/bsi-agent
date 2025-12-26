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

# Generate synthetic dialogues
python scripts/generate_dialogues.py --num_dialogues 500

# Fine-tune the model
python scripts/train.py --config configs/train_config.yaml

# Run evaluation
python scripts/evaluate.py --model_path outputs/model
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
