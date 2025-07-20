import pandas as pd
import string
import re
import nltk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import resample

# Download NLTK dependencies
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

df = pd.read_csv("reviews.csv")

# Clean pattern yang tidak dibutuhkan
def clean_review_text(text):
    pattern = r'\d+ out of \d+ found this helpful\.\s+Was this review helpful\?\s+Sign in to vote\.\s+Permalink'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

# Preprocessing dengan lemmatization
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    text = clean_review_text(str(text))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    
    filtered = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(filtered)

# Apply preprocessing
df['cleaned_review'] = df['Review_body'].apply(preprocess_text)

# Labeling: rating >= 6 = positive, else = negative
def label_sentiment(rating):
    rating = int(str(rating).split('/')[0])  # misal "8/10" → ambil angka pertama
    if rating >= 6:
        return "positive"
    else:
        return "negative"

df['label'] = df['Review Rating'].apply(label_sentiment)
df = df[df['label'].isin(['positive', 'negative'])]  # Filter hanya 2 kelas

# Balance dataset
min_count = df['label'].value_counts().min()
balanced_df = pd.concat([
    resample(df[df['label'] == label], replace=False, n_samples=min_count, random_state=42)
    for label in ['positive', 'negative']
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    balanced_df['cleaned_review'],
    balanced_df['label'],
    test_size=0.3,
    random_state=42,
    stratify=balanced_df['label']
)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_vec)

print("=== Multinomial Naive Bayes ===")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Uncomment untuk tampilkan Confusion Matrix
# cm = confusion_matrix(y_test, y_pred, labels=["positive", "negative"])
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["positive", "negative"])
# disp.plot(cmap=plt.cm.Blues)
# plt.title("Confusion Matrix - MultinomialNB")
# plt.show()

# Predict custom input
custom_message = "it's a terrible series, i don't like it at all, the plot was boring and the characters were not relatable"
custom_vector = vectorizer.transform([custom_message])
prediction = model.predict(custom_vector)

print("Predicted Sentiment:", prediction[0].capitalize())
