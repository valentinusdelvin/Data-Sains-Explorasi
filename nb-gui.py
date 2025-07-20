import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
import pandas as pd
import string, re
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import resample

# Unduh stopwords dan tokenizer
download('punkt')
download('stopwords')

# ------------------------- Text Preprocessing ------------------------- #
def clean_review_text(text):
    pattern = r'\d+ out of \d+ found this helpful\.\s+Was this review helpful\?\s+Sign in to vote\.\s+Permalink'
    cleaned_text = re.sub(pattern, '', str(text))
    return cleaned_text

def preprocess_text(text):
    text = clean_review_text(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)

def label_sentiment(rating):
    try:
        rating = int(str(rating).split('/')[0])
        return "positive" if rating >= 6 else "negative"
    except:
        return "negative"

# ------------------------- Load + Train Model ------------------------- #
def process_and_train(filepath):
    global model, vectorizer  # Agar bisa dipakai saat prediksi custom input

    df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
    df.dropna(subset=['Review_body'], inplace=True)

    df['cleaned_review'] = df['Review_body'].apply(preprocess_text)
    df['label'] = df['Review Rating'].apply(label_sentiment)

    # Balance dataset
    min_count = df['label'].value_counts().min()
    balanced_df = pd.concat([
        resample(df[df['label'] == label], replace=False, n_samples=min_count, random_state=42)
        for label in ['positive', 'negative']
    ])

    X = balanced_df['cleaned_review']
    y = balanced_df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    # Show classification report in terminal
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=["positive", "negative"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["positive", "negative"])
    disp.plot(cmap=plt.cm.Purples)
    plt.title("Confusion Matrix 💥")
    plt.tight_layout()
    plt.show()

    messagebox.showinfo("Berhasil 🎉", "Model berhasil dilatih dan siap prediksi!")

# ------------------------- Predict Custom Input ------------------------- #
def predict_custom_text():
    global model, vectorizer
    text = entry.get()
    if not text:
        messagebox.showwarning("Kosong 😅", "Masukkan teks dulu ya!")
        return
    cleaned = preprocess_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    emoji = "😃" if prediction == "positive" else "😞"
    messagebox.showinfo("Hasil Prediksi", f"Teks kamu termasuk: {prediction.capitalize()} {emoji}")

# ------------------------- GUI Layout ------------------------- #
app = ttk.Window(title="🎀 Naive Bayes Sentiment Analyzer", themename="litera", size=(600, 400))
app.resizable(False, False)

label = ttk.Label(app, text="📝 Pilih file CSV review kamu:", font=("Comic Sans MS", 14), bootstyle="info")
label.pack(pady=20)

btn_load = ttk.Button(app, text="📁 Pilih File CSV & Latih Model", bootstyle="success-outline", command=lambda: process_and_train(filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])))
btn_load.pack(pady=10)

entry_label = ttk.Label(app, text="Coba input teks kamu sendiri:", font=("Comic Sans MS", 12))
entry_label.pack(pady=10)

entry = ttk.Entry(app, width=50)
entry.pack(pady=5)

btn_predict = ttk.Button(app, text="🔍 Prediksi Sentimen", bootstyle="warning-outline", command=predict_custom_text)
btn_predict.pack(pady=10)

footer = ttk.Label(app, text="Made by Kelompok 1", font=("Arial", 9), bootstyle="secondary")
footer.pack(side="bottom", pady=15)

app.mainloop()