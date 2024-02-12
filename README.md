#Importing The Necessary Libraries
import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec
from tensorflow import keras
from keras import layers
from keras.layers import Dense, Input, Flatten
import nltk
from nltk.tokenize import word_tokenize
import html
from gensim.models import KeyedVectors  
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout,Bidirectional, LSTM
from keras import regularizers
from keras import optimizers
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
from wordcloud import WordCloud

#Reading the dataset
data=pd.read_csv(r'C:\Users\Ajitha\Desktop\testcoding\train.csv')
test_data=pd.read_csv(r'C:\Users\Ajitha\Desktop\testcoding\test.csv')

data.info()


special_characters_to_remove = r'[!‘"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~\\]'


def clean_tweet(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ').replace('\u200d', '').replace('\xa0', ' ').replace(
        '\u200c', '').replace('“', ' ').replace('”', ' ').replace('"', ' ').replace('\u200b', '')
    x = re.sub(r'\([^)]*\)', '', x)  # Remove text within parentheses
    x = re.sub('<[^<]+?>', '', x)  # Remove HTML tags
    x = re.sub(r'\d+(\.\d+)?', 'NUM ', x)  # Replace numbers with 'NUM'
    x = re.sub(special_characters_to_remove, ' ', x)  # Remove specified special characters
    x = re.sub(r'\s+', ' ', x)  # Remove extra spaces
    return x.strip()  # Strip leading/trailing spaces

data.columns
data.isnull().sum()
test_data.isnull().sum()
data=data.dropna()
data.isnull().sum()
data.duplicated().sum()
data['Tweets']
data['Tweets']=data['Tweets'].apply(processed_tweet)
data['Tweets']

import nltk
nltk.download('punkt')

def tokenize_tweets(data):
    tokenized_tweets = []
    for tweet in data["Tweets"]:
        
        cleanse_tweet = processed_tweet(tweet)
        
        
        words = word_tokenize(cleanse_tweet, language="english")
        tokenized_tweets.append(words)
    return tokenized_tweets

train_tockenised=tokenize_tweets(data)
test_data_tockenised=tokenize_tweets(test_data)

print(train_tockenised)

#printing the world_cloud
preprocessed_words = ' '.join(data['Tweets'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(preprocessed_words)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Preprocessed Text')

def extract_word_vectors(tokenized_data, vector_size=300, window=5, min_count=1, sg=0):
    model = Word2Vec(sentences=tokenized_data, vector_size=vector_size, window=window, min_count=min_count, sg=sg)
    word_vectors = {word: model.wv[word] for word in model.wv.index_to_key}
    return word_vectors

train_vectors=extract_word_vectors(train_tockenised)
test_vectors=extract_word_vectors(test_data_tockenised)

print(train_vectors)

train_labels=data['label']
test_labels=test_data['label']

# Encoding labels using LabelEncoder
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

from keras.preprocessing.text import Tokenizer


tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_tockenised)

# tokenized text to sequences of integers
X_train = tokenizer.texts_to_sequences(train_tockenised)
X_test = tokenizer.texts_to_sequences(test_data_tockenised)

# Pad sequences to have the same length
X_train = pad_sequences(X_train, maxlen=100)
X_test = pad_sequences(X_test, maxlen=100)

Model Creation
# Defining lr_schedule function
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 5:
        lr *= 0.1
    return lr


# Creating a custom LearningRateScheduler callback
class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        # Call the lr_schedule function to get the new learning rate
        new_lr = lr_schedule(epoch)
        # Set the new learning rate
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
        print(f'\nEpoch {epoch+1} Learning Rate: {new_lr}')

LSTM MOdel Creation
model = Sequential()

vocab_size = len(tokenizer.word_index) + 1
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

for word, i in tokenizer.word_index.items():
    if word in train_vectors:
        embedding_matrix[i] = train_vectors[word]

embedding_layer = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH,
                            weights=[embedding_matrix], trainable=False)
model.add(embedding_layer)

model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

optimizer = Adam(learning_rate=lr_schedule(0))
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

Model Training 

batch_size = 32
epochs = 5

lr_scheduler = CustomLearningRateScheduler()
history = model.fit(X_train, train_labels_encoded,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    callbacks=[lr_scheduler])
model.summary()

Test Accuracy

# Evaluating the model
loss, accuracy = model.evaluate(X_test, test_labels_encoded)
print(f'\nTest Accuracy (LSTM): {accuracy}')
