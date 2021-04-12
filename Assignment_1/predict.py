# ADD THE LIBRARIES YOU'LL NEED
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras import models, layers, preprocessing as kprocessing
import re
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
tokenizer = Tokenizer()
from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import Embedding
import streamlit as st
import os
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--root", type=str, help="path/")
parser.add_argument("--model", type=str, help="path/to/model.h5")
parser.add_argument("--infile", type=str, help="path/to/input.csv")
parser.add_argument("--outfile", type=str, help="path/to/output.csv")
parser.add_argument("--format", type=str, help="output format type")

# repo_root = os.path.dirname(os.path.abspath(__file__))[:os.path.dirname(os.path.abspath(__file__)).find("Assignment 1")+13]

@st.cache(allow_output_mutation=True)

def predict_on_csv(csv_path, model_name):
    df = pd.read_csv(csv_path)
    reviews = preprocess_data(df)
    padded, _ = token(reviews)
    model = load_model(model_name)
    probs = model.predict(padded)
    preds = np.argmax(probs, axis=1) + 1
    return preds, probs

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

def preprocess_data(data):
    # make all the following function calls on your data
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

def token(train_reviews):# Tokenize text

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_reviews)
    word_index = tokenizer.word_index
    vocab_size=len(word_index)
    sequences = tokenizer.texts_to_sequences(train_reviews)
    padded = pad_sequences(sequences, maxlen=29, padding='post', truncating='post')
    return padded, tokenizer

if __name__ == '__main__':

    args = parser.parse_args()
    repo_root = args.root
    model = repo_root+"/models/"+args.model
    # print(model)
    # set input image path
    input_csv_path = repo_root+"/"+args.infile
    # print(input_csv_path)
    output_csv_path = repo_root+"/"+args.outfile
    # print(output_csv_path)
    format = args.format
    preds, probs = predict_on_csv(input_csv_path, model)

    print(preds)
    print(probs)
    df = pd.read_csv(input_csv_path)
    if(format=="all"):
        df["ratings"]=''
        df["prediction probabilities"] = ''
        i=0
        for x in preds:
            df["ratings"].iloc[i] = x
            df["prediction probabilities"].iloc[i] = probs[i]
            i=i+1
    elif(format=="class"):
        df["ratings"]=''
        i=0
        for x in preds:
            df["ratings"].iloc[i] = x
            i=i+1
    else:
        df["prediction probabilities"] = ''
        i=0
        for x in preds:
            df["prediction probabilities"].iloc[i] = probs[i]
            i=i+1
    df.to_csv(output_csv_path, index=False)
    

