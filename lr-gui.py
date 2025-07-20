import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import re, string
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Unduh resource NLTK
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

def process_and_analyze(file_path):
    try:
        df = pd.read_csv(file_path, encoding='UTF-8', on_bad_lines='skip', quotechar='"')
        df.dropna(subset=['Review_body'], inplace=True)
    except Exception as e:
        messagebox.showerror("Gagal Membaca CSV", str(e))
        return

    stop_words = set(stopwords.words('english'))
    df['clean_review'] = df['Review_body'].apply(clean_text)
    df['clean_review'] = df['clean_review'].apply(lambda x: ' '.join([w for w in word_tokenize(x) if w not in stop_words]))

    sid = SentimentIntensityAnalyzer()
    df['sentiment'] = df['clean_review'].apply(lambda x: 1 if sid.polarity_scores(x)['compound'] > 0 else 0)

    X = df['clean_review']
    y = df['sentiment']
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    messagebox.showinfo("🎉 Hasil Analisis", f"Akurasi model: {accuracy*100:.2f}%")

    print("\n=== Classification Report ===\n")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg/Netral', 'Positif'], yticklabels=['Neg/Netral', 'Positif'])
    plt.title("Confusion Matrix 🧠")
    plt.xlabel("Prediksi")
    plt.ylabel("Asli")
    plt.tight_layout()
    plt.show()

    # ROC Curve
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="darkorange")
    plt.plot([0, 1], [0, 1], linestyle="--", color="navy")
    plt.title("ROC Curve 🩺")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    # Top Words
    feature_names = np.array(vectorizer.get_feature_names_out())
    coef = model.coef_[0]
    coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coef})
    top_pos = coef_df.sort_values(by='coefficient', ascending=False).head(10)
    top_neg = coef_df.sort_values(by='coefficient').head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='coefficient', y='feature', data=pd.concat([top_pos, top_neg]), palette='coolwarm')
    plt.title("🔤 Kata-kata Paling Berpengaruh")
    plt.xlabel("Koefisien")
    plt.ylabel("Kata")
    plt.tight_layout()
    plt.show()

def pilih_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        process_and_analyze(file_path)

# GUI Start
app = ttk.Window(themename="litera")
app.title("💬 Sentiment Analyzer GUI - Lucu & Kuat")
app.geometry("500x300")
app.resizable(False, False)

label = ttk.Label(app, text="📂 Pilih file review CSV kamu dulu ya!", font=("Comic Sans MS", 14), bootstyle="info")
label.pack(pady=30)

btn = ttk.Button(app, text="🔍 Mulai Analisis Sekarang", bootstyle="success-outline", command=pilih_file)
btn.pack(pady=10)

footer = ttk.Label(app, text="by Kelompok 1 😎", font=("Arial", 9), bootstyle="secondary")
footer.pack(side="bottom", pady=20)

app.mainloop()