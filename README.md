# One Task, Three Approaches: Rules vs. Machine Learning vs. Deep Learning

**Practical NLP series · Chapter 1** — a hands-on companion to Chapter 1 (*"NLP: A Primer"*) of
[*Practical Natural Language Processing*](https://www.oreilly.com/library/view/practical-natural-language/9781492054047/)
(Vajjala, Majumder, Gupta & Surana, O'Reilly 2020).

Chapter 1 explains that NLP problems can be solved with **heuristics**, **classic machine learning** or **deep learning**.
This project solves **one** task, movie-review sentiment classification on the book's **IMDB** dataset, with all three, and compares them on the same test set:
quality, cost, data needs and the kinds of mistakes each one makes.

![Accuracy comparison](figures/accuracy_comparison.png)

## Approaches

| Family | Method | Labelled data needed |
|---|---|---|
| Heuristics | Own hand-written lexicon with negation and "but" rules | 0 |
| Heuristics | VADER (NLTK), threshold tuned on the validation set | 0 |
| Machine learning | TF-IDF (unigrams + bigrams) + Multinomial Naive Bayes | 22,500 |
| Machine learning | TF-IDF (unigrams + bigrams) + Logistic Regression | 22,500 |
| Deep learning | DistilBERT fine-tuned with a plain PyTorch training loop | 4,000 on CPU / 22,500 on GPU |
| Deep learning *(domain shift)* | Ready-made DistilBERT fine-tuned on SST-2, not on IMDB | 0 |

**Dataset:** the [IMDB Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) (Maas et al., 2011), the sentiment dataset used in Chapter 4 of the book, downloaded from the same Stanford link the book uses.
50,000 balanced reviews (25,000 train / 25,000 test). Splits used here: **22,500 train**, **2,500 validation** (10% of the official training set), and a fixed, balanced sample of **5,000 test reviews** shared by all approaches (set `EVAL_TEST_SIZE = None` in `src/utils.py` to use all 25,000).
All hyper-parameters and thresholds are chosen on the validation set; the test set is used only once per model.

## Results

<!-- Paste the table from results/results_table.md here after running notebook 05 -->

*Run the notebooks to fill in this table (see "How to run").*

### Key findings
<!-- Write 3–4 bullet points in your own words after looking at your results, e.g.:
- How much better is ML than rules, and DL than ML?
- From how many labelled examples does ML beat the rule-based approaches? (learning curve)
- How much slower is DistilBERT at prediction time?
- What kinds of reviews does every approach get wrong?
-->

## Notebooks

| Notebook | Content |
|---|---|
| [01_data_exploration](notebooks/01_data_exploration.ipynb) | HTML clean-up, class balance, review lengths, Zipf's law, log-odds of words, why NLP is hard |
| [02_heuristic_approach](notebooks/02_heuristic_approach.ipynb) | Own lexicon with negation/contrast rules, coverage analysis, VADER with a tuned threshold |
| [03_machine_learning_approach](notebooks/03_machine_learning_approach.ipynb) | TF-IDF, Naive Bayes, Logistic Regression, feature weights, single-prediction explanations, learning curve |
| [04_deep_learning_approach](notebooks/04_deep_learning_approach.ipynb) | Sub-word tokenization, truncating long reviews, DistilBERT fine-tuning loop in PyTorch, domain-shift experiment |
| [05_comparison](notebooks/05_comparison.ipynb) | Accuracy, same-data comparison, cost, data needs, agreement matrix and error analysis |

## Project structure

```
practical-nlp-ch01-three-approaches/
├── notebooks/            # the five notebooks, run in order 01 → 05
├── src/
│   └── utils.py          # shared code: IMDB download + splits, tokenizer, evaluation, saving results
├── results/              # metrics (.json) and per-review predictions (.csv), created by the notebooks
├── figures/              # plots used in this README, created by the notebooks
├── requirements.txt
└── README.md
```

## How to run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/practical-nlp-ch01-three-approaches.git
cd practical-nlp-ch01-three-approaches

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate      # Windows (Git Bash)
# source .venv/bin/activate        # Linux / macOS

# 3. Install the dependencies
pip install -r requirements.txt

# 4. Start Jupyter and run the notebooks in order (01 → 05)
jupyter lab
```

The first notebook downloads the IMDB archive (about 84 MB) from Stanford and caches the reviews in `data/` (not committed to Git).

Notebook 04 fine-tunes DistilBERT on 256-token inputs. On a laptop CPU it uses a sample of 4,000 training reviews (very roughly 30–60 minutes); on an NVIDIA GPU it uses all 22,500.
Set `TRAIN_LIMIT = 500` in its configuration cell for a quick test run.

## What I learned

- **Start simple.** Rules need no data and are fully transparent, but their coverage is limited.
- **TF-IDF + Logistic Regression** is a strong, cheap, interpretable baseline that learns domain-specific words automatically.
- **Transformers** understand context and word order better, at a much higher cost in compute and interpretability.
- **Domain shift** matters: a ready-made model trained on critics' single sentences (SST-2) is not automatically as good on long IMDB user reviews.
- **Long texts** force trade-offs for transformers (truncation to 256 tokens), while TF-IDF sees the whole review.

## Tech stack

Python · pandas · scikit-learn · NLTK · PyTorch · Hugging Face Transformers & Datasets · Matplotlib · Jupyter

## Reference

Vajjala, S., Majumder, B., Gupta, A., & Surana, H. (2020). *Practical Natural Language Processing: A Comprehensive Guide to Building Real-World NLP Systems.* O'Reilly Media.
The code in this repository is my own implementation; it is not copied from the book.

## License

[MIT](LICENSE)
