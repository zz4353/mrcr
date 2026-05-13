# MRCR Mini Evaluation Setup

This repo uses a local mini split:

- `data/mini/val.jsonl`
- `data/mini/test.jsonl`

The official OpenAI MRCR setup loads chat messages, skips examples above the target
context window, calls the model, and grades with `difflib.SequenceMatcher`. The model
must start its response with `random_string_to_prepend`; otherwise the score is `0`.

## Validate The Mini Split

```powershell
python check_mini_mrcr.py
```

Expected local shape:

- `val`: 15 rows
- `test`: 75 rows
- balanced across `n_needles`, token bins, and target positions

## Install Dependencies

```powershell
pip install -r requirements.txt
```

## Configure API Key

```powershell
$env:OPENAI_API_KEY="..."
```

## Run Text Baseline

Start with `val` before spending on `test`.

```powershell
python eval_mini_mrcr.py --split val --mode text --model gpt-4.1
python eval_mini_mrcr.py --split test --mode text --model gpt-4.1 --resume
```

Outputs are saved under `runs/`.

## Run Image-History Variant

This converts older conversation turns into images and keeps the latest user turns as
text. The default keeps the latest 3 user turns in text.

```powershell
python eval_mini_mrcr.py --split val --mode image-history --model gpt-4.1 --recent-turns 3
python eval_mini_mrcr.py --split test --mode image-history --model gpt-4.1 --recent-turns 3 --resume
```

## Re-Summarize Results

```powershell
python eval_mini_mrcr.py --grade-only runs\mrcr_val_text_gpt-4.1.jsonl
```

## Useful Cheap Smoke Test

```powershell
python eval_mini_mrcr.py --split val --mode text --model gpt-4.1 --limit 1
```
