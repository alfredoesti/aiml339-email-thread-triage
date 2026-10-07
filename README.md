# Email Thread Triage: Speech-Act Classification for Contextual Summarization

Course project for AIML339 (Artificial Intelligence Project), Victoria University of Wellington, 2026.
Supervisor: Aaron Chen.

## Research question

Does classifying each email of a thread by its speech act (Request, Propose, Commit, Meeting,
Subjective, Informative) before summarizing the thread with an LLM produce better summaries than
summarizing the raw text directly?

## Pipeline

1. **Multi-label speech-act classification** (one email can carry several labels at once; 61% of
   the BC3 emails do). Two classifiers are compared on the same thread-level split:
   - Baseline: TF-IDF + `OneVsRestClassifier(LogisticRegression)`.
   - Main model: `bert-base-uncased` fine-tuned with LoRA, sigmoid outputs and class-weighted BCE loss.
2. **Contextual summarization** with an LLM (`gemini-3.8-flash`) under two conditions that share the
   same system prompt:
   - Condition A: raw thread text only.
   - Condition B: raw thread text plus the labels predicted by BERT+LoRA for each email.

   Both conditions are evaluated with ROUGE-1/2/L against the BC3 human reference summaries.
3. **Gmail triage demo** (optional, not part of the formal evaluation): an agent that labels the
   threads of a personal Gmail inbox into five categories.

## Repository structure

```
.
├── data/                              Processed BC3 data (derived from the original XML)
│   ├── bc3_emails_labeled.csv         261 emails, split column and 6 binary label columns
│   ├── bc3_sentences.csv              3,222 sentences (used for inspection only)
│   ├── bc3_summaries.csv              Human reference summaries (one set per annotator)
│   └── bc3_split.json                 Thread-level train/val/test split (seed 42)
├── src/
│   ├── bc3_parse.py                   corpus.xml + annotation.xml -> CSVs + split
│   ├── bc3_baseline_tfidf.py          TF-IDF + OneVsRest baseline (train + evaluation)
│   ├── bc3_summarize.py               LLM summaries, Conditions A and B (Gemini API)
│   ├── bc3_rouge.py                   ROUGE evaluation of Conditions A and B
│   ├── gmail_auth.py                  Gmail OAuth helper (demo)
│   └── gmail_triage_agent.py          Gmail triage agent (demo)
├── notebooks/
│   ├── bc3_bert_lora.ipynb            BERT+LoRA training and test evaluation (Google Colab, T4 GPU)
│   ├── bc3_comparison.ipynb           Baseline vs BERT+LoRA metrics and per-label F1 figure
│   ├── bc3_summarization.ipynb        Side-by-side view of the A and B summaries
│   ├── bc3_rouge.ipynb                ROUGE tables, figure and analysis
│   └── gmail_triage_demo.ipynb        Demo visualisation (outputs cleared, see Privacy)
├── scripts/
│   ├── run_triage.py                  Entry point of the Gmail triage agent (demo)
│   └── setup_gmail_labels.py          One-time creation of the Gmail labels (demo)
├── results/                           Predictions, generated summaries, scores and figures
└── requirements.txt
```

## Setup

Python 3.10+ is recommended.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The summarization step needs a Gemini API key in a `.env` file at the repository root
(this file is gitignored):

```
GEMINI_API_KEY=your-key-here
```

## Data

The project uses the **BC3 email corpus** (40 threads from the W3C mailing lists, 261 emails,
3,222 sentences) with human reference summaries and sentence-level speech-act annotations from
several annotators.

- Source: https://www.cs.ubc.ca/labs/lci/bc3/download.html
- Reference: J. Ulrich, *Supervised Machine Learning for Email Thread Summarization*, M.Sc. thesis,
  University of British Columbia, 2008. https://dx.doi.org/10.14288/1.0051210
- Licence: Creative Commons Attribution-ShareAlike 3.0. The processed CSVs in `data/` are derived
  from the corpus and are distributed under the same licence.

The processed CSVs are already included, so this step is only needed to regenerate them. Download
`corpus.xml` and `annotation.xml`, place them in `data/raw/` (gitignored), and run:

```bash
python src/bc3_parse.py
```

This script:
- aggregates the sentence-level labels of all annotators into one multi-label vector per email
  (union of annotators; emails with no speech act are labelled `Informative`);
- computes a per-email inter-annotator agreement (mean pairwise Jaccard);
- splits the data **by whole thread** with `random.Random(42)`: 28/6/6 threads for
  train/val/test, i.e. 187/37/37 emails. Emails from the same conversation never appear in two
  partitions, and the same split is used by every model.

Note: the test split contains no email with the `Informative` label, so its per-label F1 is 0.00 for
both classifiers and it lowers the macro F1 of both equally.

## Reproducing the results

Run the commands from the repository root.

| Step | Command / notebook | Output |
|------|--------------------|--------|
| 1. Baseline | `python src/bc3_baseline_tfidf.py` | `results/bc3_baseline_test_predictions.csv` |
| 2. BERT+LoRA | `notebooks/bc3_bert_lora.ipynb` on Colab (upload `data/bc3_emails_labeled.csv` to `/content/`, GPU runtime) | `results/bc3_bert_lora_test_predictions.csv` |
| 3. Classifier comparison | `notebooks/bc3_comparison.ipynb` | metrics table, `results/bc3_comparison_f1.png` |
| 4. Summaries A and B | `python src/bc3_summarize.py` | `results/bc3_summaries_condition_a.csv`, `results/bc3_summaries_condition_b.csv` |
| 5. ROUGE | `python src/bc3_rouge.py`, then `notebooks/bc3_rouge.ipynb` | `results/bc3_rouge_scores.csv`, `results/bc3_rouge_comparison.png` |

`bc3_summarize.py` skips threads that already exist in the output CSVs, so delete them first to
regenerate the summaries. Steps 3 and 5 only read the files in `results/` and make no API calls.

### Configurations behind the reported results

**TF-IDF baseline** (`src/bc3_baseline_tfidf.py`): `TfidfVectorizer(max_features=5000,
ngram_range=(1, 2), min_df=2, stop_words="english")` fitted on train only;
`LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)` in a
`OneVsRestClassifier`.

**BERT+LoRA** (`notebooks/bc3_bert_lora.ipynb`):

| Parameter | Value |
|-----------|-------|
| Base model | `bert-base-uncased` |
| LoRA | r = 8, alpha = 16, dropout = 0.1, target modules `query`, `key`, `value`, `dense` |
| Trainable parameters | 1,344,006 of 110,830,860 (1.21%) |
| Max sequence length | 256 |
| Batch size | 8 |
| Epochs | 5 |
| Optimizer | AdamW, lr = 2e-4, weight decay = 0.01, gradient clipping 1.0 |
| Scheduler | Linear with 10% warmup |
| Loss | `BCEWithLogitsLoss` with per-label `pos_weight` (train set) |
| Decision threshold | 0.5 |
| Checkpoint selection | Best validation micro F1 (epoch 5, 0.684) |
| Seed | 42 (`random`, `numpy`, `torch`) |
| Hardware | Google Colab, NVIDIA T4 GPU |

**Summarization** (`src/bc3_summarize.py`): `gemini-3.8-flash` (free tier), default sampling
parameters, one summary per thread and condition. The system prompt is the same in both conditions
(`SYSTEM_INSTRUCTION`); Condition B only adds the predicted labels to each email header plus a
short explanation of what each label means.

**ROUGE** (`src/bc3_rouge.py`): `rouge_score` with stemming, F1, computed against each annotator's
reference summary and keeping the maximum across annotators.

### Main results (test set)

Classification (37 emails, 6 threads, single run with seed 42; higher is better except Hamming loss):

| Model | Micro F1 | Macro F1 | Hamming loss |
|-------|----------|----------|--------------|
| TF-IDF baseline | 0.560 | 0.410 | 0.347 |
| BERT+LoRA | 0.637 | 0.525 | 0.369 |

Summarization (mean ROUGE F1 over the 6 test threads, one generation per thread and condition):

| Metric | Condition A | Condition B | Delta (B-A) |
|--------|-------------|-------------|-------------|
| ROUGE-1 | 0.4094 | 0.4309 | +0.0215 |
| ROUGE-2 | 0.1112 | 0.1332 | +0.0219 |
| ROUGE-L | 0.2200 | 0.2272 | +0.0072 |

Per-thread scores are in `results/bc3_rouge_scores.csv`. All numbers come from a single run
(classifiers) and a single generation per condition (summaries); see the report for the
discussion of this limitation.

## Gmail triage demo (optional)

Not part of the formal evaluation. It needs a Google Cloud OAuth client (`credentials.json` at the
repository root, gitignored) and the Gemini API key:

```bash
python scripts/setup_gmail_labels.py   # once: creates Triage/University, Work, Personal, Spam, Other
python scripts/run_triage.py           # classifies new inbox threads and applies the labels
```

The agent only sends the sender, subject and a ~300-character preview of each thread to the LLM,
and never sends or deletes email.

## Privacy and security

- No API keys, OAuth credentials or tokens are stored in the repository (`.env`,
  `credentials.json` and `token.json` are gitignored).
- The Gmail triage log (`results/gmail_triage/`) contains personal inbox data and is gitignored;
  the outputs of `notebooks/gmail_triage_demo.ipynb` were cleared for the same reason.
- BC3 contains real names and email addresses from public W3C mailing-list archives. They are kept
  as in the original corpus and used for research only.

## Author

Alfredo Estirado, Data Science and Engineering, Victoria University of Wellington.
