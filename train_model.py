import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load the dataset
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[["v1", "v2"]]
df.columns = ["label", "text"]

# convert spam/ham to 1/0
df["label"] = df["label"].map({"spam": 1, "ham": 0})

# basic text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["text"] = df["text"].apply(clean_text)

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# convert text to numbers using TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# train logistic regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# check accuracy
preds = model.predict(X_test_vec)
acc = accuracy_score(y_test, preds)
print(f"Model Accuracy: {acc * 100:.2f}%")

# save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Done! model.pkl and vectorizer.pkl saved.")
