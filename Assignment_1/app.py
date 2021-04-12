import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import base64
import uuid
import re
import random
import string
import os
import shutil
from tensorflow.keras.models import load_model
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
import streamlit as st
# from argparse import ArgumentParser


MODELS = {
	"GloVe 1 layer ReLu": "glove_1layer_relu.h5",
	# "GloVe 1 layer Softmax": "model_2", 
	"Word2Vec 1 layer ReLu": "w2vec_1layer_relu.h5", 
	# "Word2Vec 1 layer Softmax": "model_4",
	# "FastText 1 layer ReLu": "model_5",
	# "FastText 1 layer Softmax": "model_6"
}

repo_root = os.path.dirname(os.path.abspath(__file__))[:os.path.dirname(os.path.abspath(__file__)).find("Assignment_1")+13]
print(repo_root)
# Obtain the CSS for Buttons to be displayed
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_button_css(button_id):
	custom_css = f"""
        <style>
            #{button_id} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """
	
	return custom_css

# Get the raw HTML string for the button to download the created JSON file
def download_csv_button(csv):
	b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    # href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'

	button_uuid = str(uuid.uuid4()).replace('-', '')
	button_id = re.sub('\d+', '', button_uuid)

	dl_link = get_button_css(button_id) + f'<a download="output.csv" id="{button_id}" href="data:file/csv;base64,{b64}">Download CSV File</a>'
	return dl_link

# Save the uploaded dataframe temporarily
def save_file_temp(df):
	# df = pd.read_csv(uploaded_file, delimiter='\t')
	print(df.head())
	filename = "temp_" + ''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)]) + ".csv"
	df.to_csv(filename, index=False)
	return filename

def save_sent_temp(user_input):
	row = {'reviews': [user_input]}
	df = pd.DataFrame(row, columns=['reviews'])
	filename = "temp_" + ''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)]) + ".csv"
	df.to_csv(filename)
	return filename

# Check if a file exists & delete it
def delete_file(filename):
	if os.path.exists(filename):
		os.remove(filename)

# Delete the temporarily uploaded image file & it corresponsing results
def delete_temp_files(img_filename):
	delete_file(img_filename)

# Display the results (JSON & HTML) of the component detection process
def display_result(output_file, show_final, show_prob):

	download_json_btn_str
	buttons_str = f"{download_json_btn_str}{download_html_btn_str}"
	st.markdown(buttons_str, unsafe_allow_html=True)

	if show_html:
		st.write("### HTML Rendering")
		with open(html_file, 'r') as html:
			source_code = html.read()
		components.html(source_code)

	if show_json:
		st.write("### JSON Output")
		st.json(json_data)

# Submit the uploaded wireframe image for component detection
def submit_clicked(value, user_input, df, options, show_final, show_prob):
	if(value):
		with st.spinner(text='Submitted successfully. Performing sentiment analysis ...'):
			if user_input!="":
				in_filename = save_sent_temp(user_input)
			elif df.empty==False:
				in_filename = save_file_temp(df)
			else:
				st.write("Enter a sentence or upload a file")
				return
			# print(options["model"])
			# Starting prediction
			print("Running model")
			if(show_final and show_prob):
				format = "all"
			elif(show_final):
				format = "class"
			elif(show_prob):
				format = "prob"
			else:
				st.write("Choose one output option from side panel")
				return
			model_run = "python {repo_root}\predict.py --root {root} --model {model} --infile {input} --outfile {output} --format {format}".format(
				repo_root=repo_root,
				root=repo_root,
				model = options["model"],
				input=in_filename,
				output="output.csv",
				format = format
			)
			print(model_run)
			# os.system(model_run)
			model_path = repo_root+"/models/"+options["model"]
			input_csv_path = repo_root+"/"+in_filename
			output_csv_path = repo_root+"/output.csv"
			preds, probs = predict_on_csv(input_csv_path, model_path)
			df_in = pd.read_csv(input_csv_path)
			if(format=="all"):
				df_in["ratings"]=''
				df_in["prediction probabilities"] = ''
				i=0
				for x in preds:
					df_in["ratings"].iloc[i] = x
					df_in["prediction probabilities"].iloc[i] = probs[i]
					i=i+1
			elif(format=="class"):
				df_in["ratings"]=''
				i=0
				for x in preds:
					df_in["ratings"].iloc[i] = x
					i=i+1
			else:
				df_in["prediction probabilities"] = ''
				i=0
				for x in preds:
					df_in["prediction probabilities"].iloc[i] = probs[i]
					i=i+1
			df_in.to_csv(output_csv_path, index=False)
			st.success('Completed Prediction!')

			# Dummy HTML rendering of the wireframe image
			# output_file = "output.csv"

			# output = pd.read_csv(output_file)
			st.dataframe(df_in)

			# download=st.button('Download CSV File')
			# if download:
			# 	'Download Started!'
			# 	b64 = base64.b64encode(output_file.encode()).decode()  # some strings
			# 	linko= f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'
			# 	st.markdown(linko, unsafe_allow_html=True)

			delete_temp_files(in_filename)

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

def main():
	
	st.title('Sentiment Analysis')
	st.subheader('CS772 - Assignment 1')

	user_input = st.text_input("Input your sentence for sentiment analysis here", "")

	st.text("Or")
	df = pd.DataFrame()
	uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
	if uploaded_file is not None:
		# file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
		# st.write(file_details)
		df = pd.read_csv(uploaded_file)
		st.dataframe(df)

	st.sidebar.title("Outputs")

	show_final = st.sidebar.checkbox("Show Final Sentiment Class", value = True)
	show_prob = st.sidebar.checkbox("Show Probabilities of all Classes")

	st.sidebar.title("Choose Model")
	options = {}
	
	# st.sidebar.subheader("For UI Elements:")
	
	options["model"] = MODELS[st.sidebar.selectbox(
		"Model for Sentiment Analysis:", 
		options=["GloVe 1 layer ReLu", "Word2Vec 1 layer ReLu"]
	)]
	
	submitted = st.button("Submit")
	submit_clicked(submitted, user_input, df, options, show_final, show_prob)

	st.write('_Developed with ❤️ by [Shreya](https://laddhashreya2000.github.io), [Shaun]() & [Hitul]()_')

if __name__ == "__main__":
	main()