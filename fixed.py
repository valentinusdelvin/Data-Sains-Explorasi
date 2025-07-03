import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as size

text = open('reviews.csv', encoding='UTF-8').read() # Membuka Data Reviews.csv
lowercase = text.lower() # Mengubah semua kata menjadi lowercased
cleaned_word = lowercase.translate(str.maketrans('', '', string.punctuation)) # Menghilangkan semua punctuation
tokenized_word = word_tokenize(cleaned_word, "english") # Mengtokenized atau menjadikan sebuah sentence menjadi perkata

final_words = [] #List finalword
for word in tokenized_word:
    if word not in stopwords.words('english'): # Menggunakan stopword dari library
        final_words.append(word) # jika kata-kata di tokenized word tidak sesuai dengan stopword, maka akan ditambahkan ke final word
 
sid = SentimentIntensityAnalyzer() 
positive_word = []
negative_word = []
neutral_word = []

for words in final_words:
   
    value = sid.polarity_scores(words) # menilai apakah kata itu positif, negatif atau netral
    if value['compound'] > 0:
        positive_word.append(words)
    elif value['compound'] < 0:
        negative_word.append(words)
    else:
        neutral_word.append(words)

def generate_and_show_wordcloud(words_list, title):
   
    text_corpus = " ".join(words_list) # Mengubah list kata menjadi satu string tunggal, dipisahkan oleh spasi.

    
    if not text_corpus:
        print(f"Tidak ada kata untuk ditampilkan di Word Cloud '{title}'.")# Cek apakah ada kata untuk dibuatkan cloud
        return

   
    wordcloud = WordCloud(width=800, height=400, background_color="white", collocations=False).generate(text_corpus) # Membuat objek WordCloud

    # Menampilkan Word Cloud menggunakan matplotlib
    size.figure(figsize=(10, 5))
    size.imshow(wordcloud, interpolation='bilinear')
    size.axis("off")
    size.title(title, fontsize=16)  
    size.show()

# Membuat Word Cloud untuk setiap sentimen
generate_and_show_wordcloud(positive_word, "Positive Sentiment Words")
generate_and_show_wordcloud(negative_word, "Negative Sentiment Words")
generate_and_show_wordcloud(neutral_word, "Neutral Sentiment Words")