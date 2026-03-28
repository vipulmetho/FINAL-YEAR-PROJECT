# Spam Email Classification using NLP and Machine Learning
### BCA Final Year Project

---

## Project Overview

This project classifies email/SMS messages as **Spam** or **Not Spam** using:
- **TF-IDF** for converting text to numbers
- **Logistic Regression** as the classification model
- **Streamlit** for the web interface

---

## Folder Structure

```
spam_classifier/
│
├── spam.csv           ← dataset (download from Kaggle)
├── train_model.py     ← run this first to train and save the model
├── app.py             ← Streamlit web app
├── requirements.txt   ← Python dependencies
├── model.pkl          ← saved after running train_model.py
├── vectorizer.pkl     ← saved after running train_model.py
└── README.md
```

---

## How to Run Locally

### Step 1 — Set up environment

Make sure you have Python 3.10 installed. Then open your terminal/command prompt in the project folder.

```bash
pip install -r requirements.txt
```

### Step 2 — Place the dataset

Make sure `spam.csv` is in the same folder as `train_model.py`.
(The original Kaggle SMS Spam Collection dataset works directly.)

### Step 3 — Train the model

```bash
python train_model.py
```

This will create `model.pkl` and `vectorizer.pkl` in the same folder.
You should see output like:
```
Model Accuracy: 98.xx%
Done! model.pkl and vectorizer.pkl saved.
```

### Step 4 — Run the app

```bash
streamlit run app.py
```

A browser tab will open automatically at `http://localhost:8501`

---

## How to Deploy on Streamlit Cloud

1. Upload all files to a **GitHub repository** (public):
   - app.py
   - train_model.py
   - requirements.txt
   - model.pkl  ← IMPORTANT: commit this file too
   - vectorizer.pkl  ← IMPORTANT: commit this file too

2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and sign in with GitHub

3. Click **New App** → select your repository → set main file as `app.py`

4. Click **Deploy** — it will install requirements and launch the app

> ⚠️ Important: You MUST commit the `.pkl` files to GitHub before deploying.
> The Streamlit Cloud environment cannot run train_model.py during deployment.

---

## Viva Explanation Points

- **Why Logistic Regression?** Simple, fast, and works well for text classification
- **Why TF-IDF?** Converts words to numbers, gives more weight to rare/important words
- **Preprocessing:** Lowercasing and removing special characters reduces noise
- **Accuracy:** Typically 97–99% on the SMS Spam dataset
- **Pickle files:** Used to save and reload the trained model without retraining

---

## Dataset

SMS Spam Collection Dataset — available on Kaggle:
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
