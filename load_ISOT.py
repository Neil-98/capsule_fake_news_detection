# Importig standard Libraries 
import numpy as np
import pandas as pd

# Train / Test split
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Import the natural language toolkit library 
#!pip install nltk
import nltk
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords

# Text tokenization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Load dataset
d_true = pd.read_csv("True.csv")
d_fake = pd.read_csv("Fake.csv")

d_true["label"] = 1
d_fake["label"] = 0
d_true['text'] = d_true['title'] +" "+ d_true['text']
d_fake['text'] = d_fake['title'] +" "+ d_fake['text']
d_true.drop(["title", "subject", "date"], axis=1, inplace= True)
d_fake.drop(["title", "subject", "date"], axis=1, inplace= True)
data = pd.concat([d_true, d_fake], axis=0, ignore_index = True)
data = shuffle(data)

#datastructure
data = data.reset_index(drop= True)
data.head()
data.isnull().sum()
data.shape

#Cleaning stopwords
stop_words = set(stopwords.words('english'))
def process(text):
    """Converting the texts into lowercase characters and removing punctuations and stopwords using the nltk library."""
    text = text.lower()
    words = nltk.word_tokenize(text)
    new_words= [word for word in words if word.isalnum() and word not in stop_words]
    text = " ".join(new_words)
    return text
data = shuffle(data)
data['text'] = data['text'].apply(process)
X = data['text'].to_frame()
Y = data['label'].to_frame()
text_len=X['text'].str.split().map(lambda x: len(x))
Avg_len = text_len.mean()
Avg_len = round(Avg_len)
lst = []
for i in X['text']:
    tmp = i.split()
    lst.extend(tmp)
lst = set(lst)
Vocab_size = len(lst)
print("the average number of words in the texts is : ", Avg_len)
print("the texts contains", Vocab_size, "unique words")


#tokenize
tokenizer = Tokenizer(num_words=Vocab_size)
tokenizer.fit_on_texts(X['text'])
sequences = tokenizer.texts_to_sequences(X['text'])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

#Padding
data = pad_sequences(sequences, maxlen=Avg_len+2, padding='post', truncating='post')

#Spliting the data into train / test sets
X_train, X_test, Y_train, Y_test = train_test_split(data, Y, test_size=0.25, random_state=25)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

#GloVe embedding
embeddings_index = {};
with open("glove.6B.300d.txt", encoding="utf8") as f:
    for line in f:
        values = line.split();
        word = values[0];
        coefs = np.asarray(values[1:], dtype='float32');
        embeddings_index[word] = coefs;
print(len(coefs))

embeddings_matrix = np.zeros((Vocab_size+1, 300));
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word);
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector;
print(embeddings_matrix.shape)
