# ADD THE LIBRARIES YOU'LL NEED
import pandas as pd
import numpy as np
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
tokenizer = Tokenizer()
from tensorflow.keras.layers import Embedding
'''
About the task:
You are provided with a codeflow- which consists of functions to be implemented(MANDATORY).
You need to implement each of the functions mentioned below, you may add your own function parameters if needed(not to main).
Execute your code using the provided auto.py script(NO EDITS PERMITTED) as your code will be evaluated using an auto-grader.
'''


def embedding(vocab_size, word_index):
    embeddings_index = {};
    with open('/content/drive/MyDrive/CS772/glove.6B.100d.txt') as f:
        for line in f:
            values = line.split();
            word = values[0];
            coefs = np.asarray(values[1:], dtype='float32');
            embeddings_index[word] = coefs;
    # print(len(coefs))

    embeddings_matrix = np.zeros((vocab_size+1, 100));
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word);
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector;
    return embeddings_matrix


def token(train_reviews):# Tokenize text

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_reviews)
    word_index = tokenizer.word_index
    vocab_size=len(word_index)
    # print(vocab_size)
    # Padding data
    sequences = tokenizer.texts_to_sequences(train_reviews)
    padded = pad_sequences(sequences, maxlen=29, padding='post', truncating='post')
    return padded, tokenizer


#def encode_data(text):
    # This function will be used to encode the reviews using a dictionary(created using corpus vocabulary)

    # Example of encoding :"The food was fabulous but pricey" has a vocabulary of 4 words, each one has to be mapped to an integer like:
    # {'The':1,'food':2,'was':3 'fabulous':4 'but':5 'pricey':6} this vocabulary has to be created for the entire corpus and then be used to
    # encode the words into integers
    # word_index = tokenizer.word_index
    # vocab_size=len(word_index)
    # print(vocab_size)
    #INDEX = 1
    #VOCAB = {}
    # sequences = []
    # i = 0
    # for x in text:
    #   sequences.append([])
    #   for tok in x:
    #     if tok in VOCAB.keys():
    #       sequences[i].append(VOCAB[tok])
    #       continue
    #     else:
    #       VOCAB[tok] = INDEX
    #       sequences[i].append(INDEX)
    #       INDEX = INDEX+1
    #   i = i+1
    # # return encoded examples
    # print(INDEX)
    # return sequences


def convert_to_lower(text):
    # return the reviews after convering then to lowercase
    return text.str.lower()


def remove_punctuation(text):
    # return the reviews after removing punctuations
    text = text.str.replace(r'[^\w\s]', '')
    return text



def remove_stopwords(text):
    # return the reviews after removing the stopwords
    text = text.str.replace(pattern, '')
    return text

def perform_tokenization(text):
    # return the reviews after performing tokenization
    text = text.apply(lambda row: word_tokenize(row))
    return text


def perform_padding(sequences):
    # return the reviews after padding the reviews to maximum length
    # sequences = tokenizer.texts_to_sequences(data)
    # print(sequences[0])
    # print(len(sequences[0]))
    # print(sequences.shape)
    padded =  pad_sequences(sequences, padding='post', truncating='post')
    # print(padded)
    # print(padded.shape)
    return padded

def preprocess_data(data):
    # make all the following function calls on your data
    # EXAMPLE:->
    """
    review = data["reviews"]
    review = convert_to_lower(review)
    review = remove_punctuation(review)
    review = remove_stopwords(review)
    review = perform_tokenization(review)
    review = encode_data(review)
    review = perform_padding(review)
    """
    # return processed data
    review = data["reviews"]
    review = convert_to_lower(review)
    review = remove_punctuation(review)
    review = remove_stopwords(review)
    # review = perform_tokenization(review)
    # review = encode_data(review)
    # review = perform_padding(review)
    # print(review.head(5))
    return review



def softmax_activation(x):
    # write your own implementation from scratch and return softmax values(using predefined softmax is prohibited)
    exp_arr = tf.exp(x)
    return exp_arr/tf.reduce_sum(exp_arr, axis=0)


class NeuralNet:

    def __init__(self, reviews, ratings, vocab_size, matrix):

        self.reviews = reviews
        self.ratings = ratings
        self.nn_model = Sequential()
        self.vocab_size = vocab_size
        self.embeddings_matrix = matrix



    def build_nn(self):
        #add the input and output layer here; you can use either tensorflow or pytorch
        # self.nn_meodel.add(tf.keras.layers.Reshape((), input_shape=(20153,)))
        self.nn_model.add(Embedding(self.vocab_size+1, 100, weights=[self.embeddings_matrix], trainable=False, input_length=29))
        # self.nn_model.add()
        self.nn_model.add(tf.keras.layers.Flatten())
        # self.nn_model.add(tf.keras.layers.Reshape((100*,), input_shape=(100,)))
        self.nn_model.add(Dense(5, activation=softmax_activation))
        # print(self.nn_model.output_shape)
        # print(self.nn_model.input_shape)

        self.nn_model.compile(loss= tf.keras.losses.categorical_crossentropy,
                              optimizer='adam',
                              metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

    def train_nn(self,batch_size,epochs):
        # write the training loop here; you can use either tensorflow or pytorch
        # print validation accuracy
        review_val = self.reviews[:10000]
        rating_val = self.ratings[:10000]
        review_train = self.reviews[10000:]
        rating_train = self.ratings[10000:]

        callbacks = []
        callbacks.append(EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True))
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, min_lr=7.8125e-8, patience=10, min_delta=0.001, verbose=0))

        history = self.nn_model.fit(review_train, rating_train, batch_size = batch_size,
                                    epochs = epochs, validation_data = (review_val, rating_val),
                                    callbacks=callbacks, verbose=0)
        res = self.nn_model.evaluate(review_val, rating_val, verbose=0)
        print("Validation accuracy: {}%".format(round(res[1]*100, 2)))
    def predict(self, reviews):
        # return a list containing all the ratings predicted by the trained model
        results = self.nn_model.predict(reviews)
        return results


# DO NOT MODIFY MAIN FUNCTION'S PARAMETERS
def main(train_file, test_file):

    batch_size,epochs=1024, 100

    train_data = pd.read_csv(train_file, engine='python')
    test_data = pd.read_csv(test_file, engine='python')
    train_ratings = train_data['ratings']
    train_reviews = preprocess_data(train_data)
    test_reviews = preprocess_data(test_data)
    train_ratings = tf.keras.utils.to_categorical(train_ratings-1, num_classes=5)

    padded_train, tokenizer = token(train_reviews)
    word_index = tokenizer.word_index
    vocab_size=len(word_index)
    embeddings_matrix = embedding(vocab_size, word_index)
    model = NeuralNet(padded_train, train_ratings, vocab_size, embeddings_matrix)
    model.build_nn()
    try:
        model.train_nn(batch_size,epochs)
    except:
        pass
    padded_test, tokenizer = token(test_reviews)
    test_pred = model.predict(padded_test)
    return np.argmax(test_pred, axis=1) + 1
