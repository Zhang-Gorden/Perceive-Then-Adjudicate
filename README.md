# Perceive Then Adjudicate

This repository contains the source code for our paper, “Perceive-Then-Adjudicate: A Multi-Agent Framework for Resolving Knowledge-Evidence Conflicts in LLM Fact-Checking,” which presents a multi-agent framework consisting of a Knowledge Agent (KA), an Evidence Agent (EA), and a learned Adjudication Agent (AA).


## Repository Layout

```text
agents/                        Agent implementations
dataset/                       Dataset documentation (data not included)
generate_ka_ea_predictions.py  Generate KA/EA outputs and state vectors
split_predictions.py           Split JSONL records by claim identifier
evaluate_ka_ra.py              Analyze KA/EA predictions
train_aa.py                    Train the adjudication router
evaluate_aa.py                 Evaluate the adjudication router
utils.py                       Shared model and evaluation utilities
requirements.txt               Python dependencies
```


## Usage

### 1. Generate KA and EA Predictions

Using PolitiFact as an example:

```bash
python generate_ka_ea_predictions.py \
  --model-name /path/to/model \
  --data dataset/PolitiFact.csv \
  --output predictions/politifact.jsonl
```
Replace `/path/to/model` with a local model path or a compatible Hugging Face model identifier.

### 2. Analyze KA and EA Predictions

```bash
python evaluate_ka_ra.py \
  --predictions predictions/politifact.jsonl
```

### 3. Prepare Training, Validation, and Test Splits

```bash
python split_predictions.py \
  --input predictions/politifact.jsonl \
  --output-dir predictions/politifact \
  --train-ratio 0.7 \
  --val-ratio 0.1 \
  --seed 42
```

### 4. Train the Adjudication Agent

```bash
python train_aa.py \
  --model-name /path/to/model \
  --dataset dataset/PolitiFact.csv \
  --train-jsonl predictions/politifact/train.jsonl \
  --val-jsonl predictions/politifact/val.jsonl \
  --save-path checkpoints/router.pt \
  --eval-output results/validation.csv
```

### 5. Evaluate the Adjudication Agent

```bash
python evaluate_aa.py \
  --model-name /path/to/model \
  --dataset dataset/PolitiFact.csv \
  --test-jsonl predictions/politifact/test.jsonl \
  --checkpoint checkpoints/router.pt \
  --output results/test.csv
```


## Citation

If you use this code or framework in your research, please cite our paper:

```bibtex
@misc{zhang2026factchecking,
  title  = {Perceive-Then-Adjudicate: A Multi-Agent Framework for Resolving Knowledge-Evidence Conflicts in LLM Fact-Checking},
  author = {Yongcheng Zhang and Xin Zhang and Fei Li and Chong Teng and Jiangming Yang and Donghong Ji and Zhuang Li},
  year   = {2026},
  note   = {Manuscript}
}
```


## Acknowledgments

The work of the Wuhan University authors was supported by Ant Group through the CCF-Ant Research Fund. Zhuang Li received no funding for this work and is the senior author.
