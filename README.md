# Aegis: Detecting Transaction Fraud Forward in Time

**Author:** Shaivil Patel · CSCI 490-B9 Deep Learning, NIU, Fall 2026

**AI tool use:** I used Claude Code (model Claude Opus 5.5). It wrote and ran `fraud.py`, `data_check.py`, `level0_logreg.py` and `download_data.py`, adapted `AGENTS.md`, and drafted the factual parts of this README from my approved topic and the files in `results/`. It laid out options for the Level 1 and Level 2 models and the Level 3 question, and I chose among them. It also found the public work listed in section 4, drafted that comparison from the projects' public descriptions, and proposed the forecasts in section 4. No code from competition participants or other public projects was used. The observation under "What I noticed" is my own, and Claude Code cleaned up its wording.

## 1. Task

The input is one online transaction: its amount, product code, card and address fields, email domains, device and identity information when available, and Vesta's anonymized engineered features. The output is the probability that the transaction is fraud (`isFraud = 1`).

One real example from the training split, showing a few of its fields:

> **Input:** `day` = 115.9, `TransactionAmt` = 171.00, `ProductCD` = W, `card4` = visa, `card6` = credit, `P_emaildomain` = missing, `R_emaildomain` = missing, `DeviceType` = missing, `has identity data` = no, `missing` = 53%
>
> **Label:** 1 (fraud)

## 2. Motivation

A payment company or an online merchant could use this model to decide which transactions to block or send for review. It is hard because fraud is rare, only 3.52% of training transactions, so a model can look accurate while catching almost none of it. Many values are missing: 45.12% of the feature values in the training split are empty, and only 26.59% of training transactions have identity data. Several important fields are categories with thousands of possible values, and the useful signals are interactions between fields. The model must also work on transactions later than any it was trained on, while behavior changes over time.

## 3. Data

**Source.** [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) on Kaggle, with data from Vesta. I use only the two labeled files, `train_transaction.csv` joined with `train_identity.csv` on `TransactionID`, because the labels of Kaggle's test files are private.

**Size.** 590,540 transactions and 431 feature columns after the join.

**License.** The data is covered by the [Kaggle competition rules](https://www.kaggle.com/competitions/ieee-fraud-detection/rules). I do not commit or redistribute it: `ieee_data/` and CSV files are in `.gitignore`, and `download_data.py` fetches the files with the Kaggle API. `samples.md` holds only the six example rows the course requires, with a few columns.

**Labels.** According to the [organizers' data description](https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203), a transaction is labeled fraud when a chargeback is reported on the card, and later transactions linked to it by account, email or billing address are labeled fraud too.

**Split.** I sort by `TransactionDT` and use the first 70% of rows for training, the next 15% for validation and the last 15% for test. The cuts are made at timestamps, so transactions with the same timestamp stay together. A deployed model predicts future fraud from past fraud, so training must come before validation and validation before test. The split is implemented once, in `fraud.load_splits()`.

Output of `data_check.py` (saved in `results/data_check.json`):

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| Not fraud (0) | 398,840 | 85,539 | 85,498 |
| Fraud (1) | 14,538 | 3,042 | 3,083 |
| **Total** | **413,378** | **88,581** | **88,581** |
| Fraud rate | 3.52% | 3.43% | 3.48% |
| Days covered | 1.0 to 120.8 | 120.8 to 152.2 | 152.2 to 183.0 |
| Rows with identity data | 26.59% | 17.67% | 21.07% |

The data has no card id, so a card can appear in more than one split. That matches deployment, where returning customers are normal. Using the six card columns as an approximate card key, 98.66% of validation rows and 98.05% of test rows have a key already seen in training. My Level 3 question measures what this overlap is worth.

Three examples of each class are in [samples.md](samples.md).

**What I noticed.** All three fraud samples have higher transaction amounts than all three non-fraud samples. All three fraud samples used credit cards, while one of the three non-fraud samples used a debit card.

**From the data check.** In the full training split, `data_check.py` reports a median amount of 75.00 for fraud and 68.95 for not fraud.

## 4. Method

**Level 0 result** (from `results/level0_logreg_val.json` and `results/level0_logreg_test.json`). The model is logistic regression with class weighting. Its 382 numeric columns are median-imputed and standardized, and its 33 categorical columns with at most 64 values are one-hot encoded, all fitted on the training split. The 16 categorical columns with more values are left out at this level. The settings are fixed, not tuned.

| Model | Split | PR-AUC | ROC-AUC | Precision at the threshold | Recall at the threshold | PR-AUC of random guessing |
|---|---|---:|---:|---:|---:|---:|
| Level 0: logistic regression | Validation | 0.410 | 0.848 | 0.500 | 0.363 | 0.034 |
| Level 0: logistic regression | Test | 0.187 | 0.832 | 0.296 | 0.367 | 0.035 |

The threshold is the one that reaches precision 0.50 on validation, and it is reused unchanged on test.

**Plan for the next levels.** The numbers below are my forecasts, not results.

| Level | Model | Expected PR-AUC on validation |
|---|---|---:|
| 1 | PyTorch MLP with learned embeddings for categorical columns, plus my own history features | 0.50 |
| 2 | TabPFN, a pretrained tabular Transformer, fine-tuned on a subsample of rows and a reduced feature set | 0.42 |
| Reference | LightGBM on the same features as Level 1, as a strong tabular comparison (not a level) | 0.55 |
| 3 | The question below | – |

*Level 1 features.* For each transaction, the history features describe what happened earlier on the same approximate card and the same purchaser email: how many earlier transactions there were, how long ago the last one was, and how this amount compares with the earlier amounts. They are computed in time order from earlier transactions only and never use labels, so the same code could run when a transaction arrives.

*Level 2.* Following my professor's suggestion, I replaced DistilBERT on serialized text with a Transformer built for tables. I will compare [TabPFN](https://github.com/PriorLabs/TabPFN) before and after fine-tuning, so the effect of fine-tuning can be seen on its own. Each run must stay under the 30-minute limit on a Colab T4.

**Level 3 question:** How much of the advantage of a random split comes from seeing the same cards in training and evaluation, and how much comes from ignoring change over time? I will train the same model on three splits: a random split, a random split that keeps each approximate card on one side, and the time split. Comparing PR-AUC across them separates the two effects: random vs card-disjoint measures repeat cards, and card-disjoint vs time measures change over time. I will also compare the time-split score on cards seen and not seen in training.

### Closest public solutions, and how this project differs

IEEE-CIS has many public solutions. I list the closest ones I know of from their public descriptions. I did not use their code, features, settings or scores.

| Public work | What it describes | How this project differs |
|---|---|---|
| 1st place team, FraudSquad ([write-up](https://www.kaggle.com/competitions/ieee-fraud-detection/writeups/fraudsquad-1st-place-solution-part-2), [client-ID explanation](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111453), [NVIDIA summary](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution)) | Client IDs built from card, address and time fields, per-client aggregate features, and a blend of gradient-boosted tree models, aimed at Kaggle's hidden test set | I evaluate on a held-out future part of the labeled file, my history features use only earlier transactions, and my main models are a neural network and a pretrained Transformer |
| [ConlerSimmons/DSC_599_Semester_Project](https://github.com/ConlerSimmons/DSC_599_Semester_Project) | TabTransformer compared with a graph network on a 70/15/15 time split, with PR-AUC | Similar split and metric. I use an embedding MLP with past-only features, and a pretrained Transformer that is fine-tuned rather than trained from scratch |
| [MuathLoss/FT-Tabular-with-GraphSage-R-GCN-HAN-GAT-](https://github.com/MuathLoss/FT-Tabular-with-GraphSage-R-GCN-HAN-GAT-) | FT-Transformer combined with graph neural networks | I use no graph model, and my Transformer starts from pretrained weights |
| [aakashshahani/fraudguard](https://github.com/aakashshahani/fraudguard) | A leakage-focused pipeline with a chronological 70/15/15 split and PR-AUC | Similar evaluation design. My contribution is in the neural models and the Level 3 experiment |
| [lflsphere/ieee-fraud-detection](https://github.com/lflsphere/ieee-fraud-detection) | Chronological model comparison that measures how much a random split inflates PR-AUC and reports repeat-fraud entities | This is the closest work to my Level 3 question. I add a card-disjoint split to separate repeat cards from change over time |
| [knsiuss/ieee-fraud-detection](https://github.com/knsiuss/ieee-fraud-detection) | Compares random and time-ordered evaluation | Same starting point for Level 3. I break the gap into two parts and report PR-AUC as the primary metric |

## 5. Evaluation

**Metric.** Fraud is a rare positive class, so the primary metric is **PR-AUC**, which measures how well fraud is ranked at the top and whose random-guessing value equals the fraud rate. I also report **ROC-AUC**, and **precision and recall at the threshold** that reaches precision 0.50 on validation, which describes a review queue where at least half of the flagged transactions are fraud on validation. Accuracy does not fit: a model that never flags fraud would be right on every "Not fraud" row in the table above.

**Splits.** All choices (features, models, hyperparameters, epochs and the threshold) are made on validation. Each final model is evaluated once on test.

**Success.** Level 1 succeeds if its mean validation PR-AUC over seeds 0, 1 and 2 beats Level 0 by more than the standard deviation across those seeds. Level 2 succeeds if fine-tuning improves TabPFN's validation PR-AUC by more than that spread. A negative result still counts if the comparison is fair and I can explain it.

## 6. Plan

| Week | What I will do |
|---|---|
| 11/17 | Data check, Level 0 and this proposal |
| 11/18 – 11/22 | Build the past-only history features. Train the Level 1 MLP and choose its settings on validation. Train the LightGBM reference |
| 11/23 – 11/26 | Run Level 1 with three seeds, evaluate on test once, make the loss figure, and submit the progress report by the latest date, 11/26 |
| 11/27 – 11/30 | Run TabPFN before and after fine-tuning within the 30-minute limit, choose settings on validation, and evaluate on test once |
| 12/1 – 12/3 | Present Levels 0 to 2 in class |
| After 12/3 | Run the three-split Level 3 experiment and write the final report |

**Level 3 question:** how much of a random split's advantage comes from repeat cards, and how much from ignoring change over time? The evidence is PR-AUC from the same model on the random, card-disjoint and time splits, plus the time-split score on seen and unseen cards.

## 7. Progress report (Level 1)

This section will be completed with the progress report.

## Results

| Level | Model | Validation PR-AUC | Test PR-AUC | Results file |
|---|---|---:|---:|---|
| 0 | Logistic regression | 0.410 | 0.187 | `results/level0_logreg_val.json`, `results/level0_logreg_test.json` |

## How to run

Use 64-bit Python 3.11. The results here used numpy 1.26.4, pandas 2.1.4 and scikit-learn 1.6.1.

```
pip install -r requirements.txt
python download_data.py    # needs a Kaggle API token and accepted competition rules
python data_check.py       # prints the table above, writes samples.md and results/data_check.json
python level0_logreg.py    # writes the two Level 0 results files
```

`mustcite.py` and `data/` come from the course template and are left unchanged. This project uses only `save_results` from `mustcite.py`.

## Data source

IEEE-CIS Fraud Detection, Kaggle competition data provided by Vesta: https://www.kaggle.com/competitions/ieee-fraud-detection
