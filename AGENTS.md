**Task (own task, approved):** binary classification of online transactions on IEEE-CIS: from one transaction's features, predict the probability that it is fraud (`isFraud = 1`).

# Rules for coding agents in this repository

This is a student course project (CSCI 490-B9 Deep Learning, NIU, Fall 2026). Read this file at the start
of every session and follow it. If a request conflicts with these rules, say so and do not do it.

## The project

- Data: the labeled IEEE-CIS Kaggle files `train_transaction.csv` and `train_identity.csv` in `ieee_data/`.
  Load them only through the functions in `fraud.py`.
- The Kaggle data must never be committed or redistributed. `ieee_data/` and CSV files stay in `.gitignore`.
  `samples.md` and the README example are the only excerpts, and they stay as small as the course requires.

## Independent work

- Do not use code, notebooks, feature recipes, hyperparameters or scores from competition participants or other
  public projects as a foundation, and do not open their code.
- Kaggle's data pages and the organizers' data discussions may be used to understand the fields and the labels.
- Public solutions are named in the README only to say how this project differs, as the professor asked.

## Data and split

- The split is fixed by time on `TransactionDT`: the first 70% of rows are train, the next 15% validation, the last 15% test.
  Transactions with the same timestamp stay in the same split. `fraud.load_splits()` is the only way to get it.
- Never re-split, shuffle across splits, or move rows between splits. Never train on validation or test rows.
  The Level 3 experiment may build other splits, only for that question, in its own script.
- Fit imputers, scalers, encoders and feature statistics on the training split only.
- History features may use only transactions earlier than the current one.
- Do not edit `mustcite.py` or the files in `data/`. They come from the course template. Only `save_results` is used.

## Metrics

- Score every model with `fraud.evaluate`: PR-AUC (primary), ROC-AUC, and precision and recall at the threshold
  chosen for precision 0.50. Never report accuracy alone: predicting "not fraud" for everything is about 96.5% accurate.

## The test set is used once

- Choose models, hyperparameters, epochs and thresholds on the validation set only.
- Evaluate a model on the test set once, after all choices are made. Never use a test number to change anything.

## Every number comes from a file

- Save the metrics of every run with `save_results(name, metrics, config)`; it writes `results/<name>.json`.
- Every number in `README.md` must come from a file in `results/`. Never type a number that is not in a file.
  Expected results in the plan are forecasts and must be labeled as forecasts.

## Code

- Small, readable scripts, one per model or experiment, named like `level1_mlp.py`. Comment what is not obvious.
- One training run must take less than 30 minutes on a free Colab T4 GPU.
- Fix random seeds, and save the settings of each run in the `config` of `save_results`.
- Do not commit model weights or other large files (see `.gitignore`).
- After each working step, suggest a commit with a message that says what changed.

## The student is the author

- The student must be able to explain every line. Explain what you wrote and why, briefly.
- Do not write the README's analysis or conclusions; the student writes those from the results.
