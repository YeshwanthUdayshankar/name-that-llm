# Name That LLM: TF-IDF (word 1-2 + char 1-5) + LinearSVC baseline.
# Local prompt-grouped 5-fold CV: balanced accuracy 0.334.
import os
from pathlib import Path

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

D = Path(os.environ.get("NTL_DATA", "/kaggle/input/competitions/name-that-llm"))
train = pd.read_csv(D / "train.csv")
test = pd.read_csv(D / "test.csv")

word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=False,
                       token_pattern=r"(?u)\b\w+\b|[^\w\s]", min_df=2,
                       sublinear_tf=True, max_features=200_000)
char = TfidfVectorizer(analyzer="char", ngram_range=(1, 5), lowercase=False,
                       min_df=2, sublinear_tf=True, max_features=400_000)
X = hstack([word.fit_transform(train.text), char.fit_transform(train.text)]).tocsr()
clf = LinearSVC(C=0.5).fit(X, train.model)
Xt = hstack([word.transform(test.text), char.transform(test.text)]).tocsr()

sub = pd.DataFrame({"id": test["id"], "model": clf.predict(Xt)})
sub.to_parquet("submission.parquet", index=False)
print(sub.shape, sub.model.value_counts().head())
