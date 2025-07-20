import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# --- Persiapan Awal NLTK (Mencoba mengunduh jika belum ada) ---
try:
    stopwords.words('english')
except LookupError:
    print("Mengunduh paket 'stopwords' NLTK...")
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Mengunduh paket 'punkt' NLTK...")
    nltk.download('punkt')


# --- 1. Memuat dan Membersihkan Data ---
try:
    df = pd.read_csv('reviews.csv', encoding='UTF-8', on_bad_lines='skip', quotechar='"')
    df.dropna(subset=['Review_body', 'Review Rating'], inplace=True)
except FileNotFoundError:
    print("Error: File 'reviews.csv' tidak ditemukan.")
    exit()

# --- 2. Membuat Label (y) dari 'Review Rating' ---
df.loc[:, 'numeric_rating'] = df['Review Rating'].apply(lambda x: int(str(x).split('/')[0]))
df = df[~df['numeric_rating'].isin([5, 6])]
df['sentiment'] = df['numeric_rating'].apply(lambda rating: 1 if rating > 6 else 0)

print("Distribusi Kelas Sentimen (setelah filtering):")
print(df['sentiment'].value_counts())
print("-" * 40)

# --- 3. Membersihkan dan Memproses Teks ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

df['clean_review'] = df['Review_body'].apply(clean_text)

# Menggunakan stopwords dari library NLTK
stop_words = set(stopwords.words('english'))
df['clean_review'] = df['clean_review'].apply(
    lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words])
)


# --- 4. Membuat Fitur (X) dan Label (y) ---
X = df['clean_review']
y = df['sentiment']

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(X)

# --- 5. Melatih Model ---
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)
model = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# --- 6. Mengevaluasi Model ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAkurasi Model: {accuracy * 100:.2f}%\n")
print("Laporan Klasifikasi:\n", classification_report(y_test, y_pred, target_names=['Negatif', 'Positif']))

# --- 7. Visualisasi Hasil (Menampilkan Langsung) ---
print("\nMenampilkan hasil visualisasi...")

# a. Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'])
plt.xlabel('Prediksi Label')
plt.ylabel('Label Sebenarnya')
plt.title('Confusion Matrix Heatmap')
plt.show()

# b. ROC Curve
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Kurva ROC (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Kurva Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# c. Fitur Paling Penting
feature_names = np.array(vectorizer.get_feature_names_out())
coefficients = model.coef_[0]
coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefficients})
top_positive = coef_df.sort_values(by='coefficient', ascending=False).head(15)
top_negative = coef_df.sort_values(by='coefficient', ascending=True).head(15)
top_features = pd.concat([top_positive, top_negative])
plt.figure(figsize=(12, 10))
sns.barplot(x='coefficient', y='feature', data=top_features, palette="RdBu_r")
plt.title('Fitur Paling Penting untuk Prediksi Sentimen')
plt.xlabel('Nilai Koefisien (Pentingnya)')
plt.ylabel('Fitur (Kata/Frasa)')
plt.tight_layout()
plt.show()
