| Approach                                  |   Accuracy |   Macro F1 |   Labelled IMDB examples |   Training time (s) |
|:------------------------------------------|-----------:|-----------:|-------------------------:|--------------------:|
| VADER (rule-based)                        |     0.7106 |     0.7077 |                        0 |                0    |
| Own lexicon + rules                       |     0.7332 |     0.721  |                        0 |                0    |
| TF-IDF + Naive Bayes                      |     0.8618 |     0.8616 |                    22500 |                0.09 |
| TF-IDF + Logistic Regression              |     0.9038 |     0.9038 |                    22500 |                1.67 |
| Ready-made SST-2 model (no IMDB training) |     0.8802 |     0.8801 |                        0 |                0    |
| DistilBERT fine-tuned on IMDB             |     0.8972 |     0.8972 |                     4000 |             2670.63 |