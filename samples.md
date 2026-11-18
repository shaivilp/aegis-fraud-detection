# Sample transactions

Three transactions of each class, drawn at random (seed 0) from the training split by `data_check.py`.
Only a few readable columns are shown. The full data is not in this repository.
`day` counts days from Kaggle's reference point. `missing` is the share of this row's features with no value.

| Label | day | TransactionAmt | ProductCD | card4 | card6 | P_emaildomain | R_emaildomain | DeviceType | has identity data | missing |
|---|---:|---|---|---|---|---|---|---|---|---:|
| 0 (not fraud) | 116.8 | 49.00 | W | visa | debit | yahoo.com | missing | missing | no | 49% |
| 0 (not fraud) | 54.3 | 59.00 | W | mastercard | credit | gmail.com | missing | missing | no | 53% |
| 0 (not fraud) | 11.0 | 100.00 | H | american express | credit | gmail.com | missing | desktop | yes | 30% |
| 1 (fraud) | 115.9 | 171.00 | W | visa | credit | missing | missing | missing | no | 53% |
| 1 (fraud) | 20.7 | 335.00 | W | visa | credit | yahoo.com | missing | missing | no | 49% |
| 1 (fraud) | 46.9 | 149.25 | C | visa | credit | gmail.com | gmail.com | desktop | yes | 20% |
