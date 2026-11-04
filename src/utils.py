"""
utils.py - Shared helper functions used by every notebook in this project.

Why a separate file?
--------------------
All five notebooks need to do the same three things:
    1. load the same dataset,
    2. evaluate predictions in exactly the same way,
    3. save their results so the final comparison notebook can read them.

If each notebook had its own copy of that code, a small difference (for
example, one notebook computing F1 differently) would make the comparison
unfair. Keeping the logic in ONE place guarantees that every approach is
measured with the same ruler. This is also how real projects are organised:
notebooks for exploration and storytelling, .py files for reusable code.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import json                      # to save results as human-readable .json files
import re                        # regular expressions, used by the tokenizer
import time                      # to measure how long training / prediction takes
from pathlib import Path         # modern, OS-independent way to handle file paths

import pandas as pd              # tables (DataFrames) for the text + label data
from sklearn.metrics import (    # standard evaluation metrics from scikit-learn
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
# __file__ is the path of THIS file (src/utils.py).
# .resolve() turns it into an absolute path, .parents[1] goes up two levels:
#     src/utils.py -> src/ -> project root
# Using paths relative to the project root means the code works no matter
# which folder you start Jupyter from (Windows, Linux or macOS).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"                 # cached copy of the dataset
RESULTS_DIR = PROJECT_ROOT / "results"           # metrics of each approach (.json)
PREDICTIONS_DIR = RESULTS_DIR / "predictions"    # per-review predictions (.csv)
FIGURES_DIR = PROJECT_ROOT / "figures"           # saved plots for the README

# Human-readable names for the two classes. In this dataset 0 = negative, 1 = positive.
LABEL_NAMES = {0: "negative", 1: "positive"}

# The IMDB "Large Movie Review Dataset" (Maas et al., 2011): the sentiment dataset
# used in Chapter 4 of the book. This is the same download link the book uses.
IMDB_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

# The same dataset is mirrored on the Hugging Face Hub. Used only if the
# Stanford server cannot be reached.
HF_MIRROR = "stanfordnlp/imdb"

# The official split is 25,000 training and 25,000 test reviews.
# * VALIDATION_SHARE: share of the official TRAINING reviews we set aside for
#   tuning (thresholds, hyper-parameters). The test set is never used for tuning.
# * EVAL_TEST_SIZE: how many of the 25,000 test reviews we evaluate on. Every
#   approach uses exactly the same reviews. 5,000 keeps the deep-learning
#   notebook manageable on a laptop CPU. Set it to None to use all 25,000.
VALIDATION_SHARE = 0.10
EVAL_TEST_SIZE = 5000
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Loading the data
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Minimal cleaning of an IMDB review.

    The raw reviews contain HTML line breaks such as "<br /><br />" (they come
    from a website). We replace every HTML tag with a space and collapse
    repeated whitespace. Nothing else is changed: capital letters and
    punctuation are kept, because some approaches (e.g. VADER) use them.
    """
    text = re.sub(r"<[^>]+>", " ", text)        # remove HTML tags like <br />
    return re.sub(r"\s+", " ", text).strip()     # collapse multiple spaces


def _download_from_stanford() -> pd.DataFrame:
    """
    Download aclImdb_v1.tar.gz (about 84 MB) from Stanford, exactly like the
    book does, and read the 50,000 labelled reviews into one DataFrame.

    Inside the archive, every review is a separate small text file:
        aclImdb/train/pos/*.txt   aclImdb/train/neg/*.txt
        aclImdb/test/pos/*.txt    aclImdb/test/neg/*.txt
    (aclImdb/train/unsup/ holds 50,000 unlabelled reviews, which we skip.)
    """
    import tarfile
    import urllib.request

    archive = DATA_DIR / "aclImdb_v1.tar.gz"
    print(f"Downloading {IMDB_URL} (about 84 MB, this can take a few minutes) ...")
    urllib.request.urlretrieve(IMDB_URL, archive)

    rows = []
    # Read the files directly from the compressed archive. This is faster than
    # extracting 100,000 small files to disk first, especially on Windows.
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            parts = member.name.split("/")      # e.g. ['aclImdb', 'train', 'pos', '0_9.txt']
            if (member.isfile() and len(parts) == 4 and parts[1] in ("train", "test")
                    and parts[2] in ("pos", "neg") and parts[3].endswith(".txt")):
                text = tar.extractfile(member).read().decode("utf-8")
                rows.append({
                    "split": parts[1],
                    "text": text,
                    "label": 1 if parts[2] == "pos" else 0,
                })
    archive.unlink()                            # delete the archive; the CSV cache is enough
    return pd.DataFrame(rows)


def _download_from_hf_mirror() -> pd.DataFrame:
    """Fallback: the same IMDB reviews from the Hugging Face Hub."""
    from datasets import load_dataset          # imported here: only needed for the fallback

    print(f"Downloading the mirror '{HF_MIRROR}' from the Hugging Face Hub ...")
    hf = load_dataset(HF_MIRROR)
    frames = []
    for split in ("train", "test"):
        df = hf[split].to_pandas()[["text", "label"]]
        df["split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_imdb() -> dict:
    """
    Load the IMDB movie-review dataset as three pandas DataFrames.

    Returns
    -------
    dict with keys "train", "validation", "test".
    Each value is a DataFrame with two columns:
        text  : the cleaned review (str)
        label : 0 = negative, 1 = positive (int)

    How it works
    ------------
    * The FIRST call downloads the original archive from Stanford, cleans the
      text, and saves two CSV files in data/ (official train and test reviews).
    * Every LATER call reads those CSV files: fast, and works offline.
    * Then the validation set is split off from train, and the test sample is
      drawn. Both use a fixed random seed, so every notebook and every run gets
      exactly the same reviews in each split.
    """
    train_csv, test_csv = DATA_DIR / "imdb_train.csv", DATA_DIR / "imdb_test.csv"

    if not (train_csv.exists() and test_csv.exists()):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            df = _download_from_stanford()
        except Exception as error:              # e.g. no connection to Stanford
            print(f"Stanford download failed ({error}). Trying the mirror ...")
            df = _download_from_hf_mirror()
        df["text"] = df["text"].apply(clean_text)
        df[df["split"] == "train"][["text", "label"]].to_csv(train_csv, index=False)
        df[df["split"] == "test"][["text", "label"]].to_csv(test_csv, index=False)
        print(f"Saved a local copy to: {DATA_DIR}")

    full_train = pd.read_csv(train_csv)
    full_test = pd.read_csv(test_csv)

    # Split the official training set into train + validation.
    # stratify=... keeps the 50/50 class balance in both parts.
    from sklearn.model_selection import train_test_split
    train, val = train_test_split(
        full_train, test_size=VALIDATION_SHARE,
        stratify=full_train["label"], random_state=RANDOM_STATE,
    )

    # Draw a fixed, balanced sample of the test set (or keep all of it).
    if EVAL_TEST_SIZE is not None and EVAL_TEST_SIZE < len(full_test):
        test, _ = train_test_split(
            full_test, train_size=EVAL_TEST_SIZE,
            stratify=full_test["label"], random_state=RANDOM_STATE,
        )
    else:
        test = full_test

    # reset_index(drop=True) numbers the rows 0, 1, 2, ... again after the shuffling
    return {
        "train": train.reset_index(drop=True),
        "validation": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


# ---------------------------------------------------------------------------
# 1b. A very simple tokenizer
# ---------------------------------------------------------------------------
# A "token" is one unit of text, here roughly one word.
# The regular expression below reads as:
#     [a-z0-9]+        one or more letters or digits      -> "movie", "2002"
#     (?:'[a-z]+)?     optionally followed by ' + letters  -> "it's", "n't"
# Everything else (spaces, commas, quotes, dots ...) is thrown away.
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def simple_tokenize(text: str) -> list:
    """
    Split a text into lowercase word tokens.

    >>> simple_tokenize("It isn't a GREAT film, but it's fun!")
    ['it', "isn't", 'a', 'great', 'film', 'but', "it's", 'fun']
    """
    # str(...) protects against missing values; the curly apostrophe (’) is
    # replaced with a normal one (') so both spellings are treated the same.
    text = str(text).lower().replace("’", "'")
    return TOKEN_PATTERN.findall(text)


# ---------------------------------------------------------------------------
# 2. Evaluating predictions
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred) -> dict:
    """
    Compare true labels with predicted labels and return the main metrics.

    accuracy  : share of reviews classified correctly.
    precision : of all reviews predicted as class X, how many really are X.
    recall    : of all reviews that really are X, how many we found.
    f1        : harmonic mean of precision and recall.

    We use the *macro* average: compute each metric per class, then take the
    simple mean. This treats both classes as equally important.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision), 4),
        "recall_macro": round(float(recall), 4),
        "f1_macro": round(float(f1), 4),
        # Rows = true class, columns = predicted class, order [negative, positive].
        # .tolist() converts the NumPy array into a plain list so JSON can store it.
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def print_report(name: str, metrics: dict) -> None:
    """Pretty-print the metrics dictionary returned by `evaluate`."""
    print(f"=== {name} ===")
    for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        print(f"{key:<16}: {metrics[key]:.4f}")
    (tn, fp), (fn, tp) = metrics["confusion_matrix"]
    print("confusion matrix (rows = true, columns = predicted):")
    print(f"               pred_neg  pred_pos")
    print(f"  true_neg     {tn:>8}  {fp:>8}")
    print(f"  true_pos     {fn:>8}  {tp:>8}")


# ---------------------------------------------------------------------------
# 3. Saving results so notebook 05 can compare all approaches
# ---------------------------------------------------------------------------
def save_results(approach: str, metrics: dict, **extra) -> Path:
    """
    Save the metrics of one approach to results/<approach>.json.

    `**extra` collects any additional keyword arguments, e.g.
        save_results("logreg", metrics, train_seconds=1.2, n_train_examples=8530)
    so each notebook can store whatever extra information is relevant.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    record = {"approach": approach, **metrics, **extra}
    path = RESULTS_DIR / f"{approach}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)            # indent=2 -> readable file
    print(f"Results saved to {path.relative_to(PROJECT_ROOT)}")
    return path


def save_predictions(approach: str, texts, y_true, y_pred) -> Path:
    """
    Save every test review with its true and predicted label to
    results/predictions/<approach>.csv. Notebook 05 uses these files for
    error analysis (which reviews does each approach get wrong?).
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"text": list(texts), "label": list(y_true), "pred": list(y_pred)})
    path = PREDICTIONS_DIR / f"{approach}.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# 4. A tiny timer
# ---------------------------------------------------------------------------
class Timer:
    """
    Measure how long a block of code takes, using Python's `with` statement:

        with Timer() as t:
            model.fit(X, y)
        print(t.seconds)

    `__enter__` runs when the `with` block starts, `__exit__` when it ends.
    """

    def __enter__(self):
        self.start = time.perf_counter()   # high-precision clock
        return self

    def __exit__(self, *args):
        self.seconds = round(time.perf_counter() - self.start, 3)
