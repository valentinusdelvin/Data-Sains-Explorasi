import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# --- Persiapan Awal NLTK ---
# Pastikan data NLTK sudah diunduh. Jalankan sekali jika belum.
# nltk.download('punkt')
# nltk.download('stopwords')

# --- 1. Memuat dan Membersihkan Data dengan Pandas ---
try:
    # Membaca CSV dengan Pandas agar setiap ulasan menjadi satu baris
    df = pd.read_csv('reviews.csv', encoding='UTF-8', on_bad_lines='skip', quotechar='"')
    # Menghapus baris yang tidak memiliki ulasan
    df.dropna(subset=['Review_body'], inplace=True)
except FileNotFoundError:
    print("Error: File 'reviews.csv' tidak ditemukan.")
    exit()

# Fungsi untuk membersihkan dan menormalisasi teks
def clean_text(text):
    text = str(text).lower()  # Lowercasing
    text = re.sub(r'\d+', '', text)  # Menghapus angka
    text = text.translate(str.maketrans('', '', string.punctuation))  # Menghapus pungtuasi
    text = text.strip()  # Menghapus spasi di awal/akhir
    return text

# Menerapkan pembersihan pada kolom ulasan
df['clean_review'] = df['Review_body'].apply(clean_text)

# Menghapus stopwords
stop_words = set(stopwords.words('english'))
df['clean_review'] = df['clean_review'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))

# --- 2. Membuat Fitur (X) dan Label (y) ---

# Inisialisasi VADER
sid = SentimentIntensityAnalyzer()

# Membuat label sentimen (y) berdasarkan skor compound dari VADER untuk setiap ulasan
# 1 untuk positif, 0 untuk negatif/netral
df['sentiment'] = df['clean_review'].apply(lambda x: 1 if sid.polarity_scores(x)['compound'] > 0 else 0)

# Menetapkan X dan y
X = df['clean_review']
y = df['sentiment']

# --- 3. Vektorisasi Teks ---
# Mengubah teks menjadi matriks numerik menggunakan TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# --- 4. Melatih Model Logistic Regression ---

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

# Melatih model
# Note: StandardScaler tidak diperlukan (dan tidak bekerja dengan baik) pada data sparse TF-IDF
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

# --- 5. Mengevaluasi Model ---

# Memprediksi data uji
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred))

# --- 6. Visualisasi Hasil ---

# a. Confusion Matrix dengan Heatmap
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negatif/Netral', 'Positif'], 
            yticklabels=['Negatif/Netral', 'Positif'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.show()

# b. ROC Curve
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# c. Visualisasi Kata Paling Penting (Fitur)
# Mendapatkan nama fitur (kata) dari vectorizer
feature_names = np.array(vectorizer.get_feature_names_out())
# Mendapatkan koefisien dari model
coefficients = model.coef_[0]

# Membuat DataFrame untuk koefisien
coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefficients})

# Mengurutkan berdasarkan koefisien
top_positive = coef_df.sort_values(by='coefficient', ascending=False).head(15)
top_negative = coef_df.sort_values(by='coefficient', ascending=True).head(15)

plt.figure(figsize=(12, 8))
sns.barplot(x='coefficient', y='feature', data=pd.concat([top_positive, top_negative]), palette="RdBu")
plt.title('Kata Paling Penting untuk Prediksi Sentimen')
plt.xlabel('Nilai Koefisien (Pentingnya)')
plt.ylabel('Kata (Fitur)')
plt.show()