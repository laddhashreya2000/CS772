import streamlit as st
import streamlit.components.v1 as components
# from lime_explainer import explainer, tokenizer, METHODS
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
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras import model`s, layers, preprocessing as kprocessing
from transformers import TFBertModel,  BertConfig, BertTokenizerFast
import streamlit as st
# from argparse import ArgumentParser
import lime
from lime.lime_text import LimeTextExplainer

MODELS = {
    "BERT": "model_noprocess.h5"
}
model_name = 'bert-base-uncased'

# Load transformers config and set output_hidden_states to False
config = BertConfig.from_pretrained(model_name)
config.output_hidden_states = False

# Load BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)
repo_root = os.path.dirname(os.path.abspath(__file__))[:os.path.dirname(os.path.abspath(__file__)).find("Assignment_1")+13]
import_model = load_model(repo_root+"/models/model_noprocess.h5")
class_names = ['1', '2', '3', '4', '5']
explainer = LimeTextExplainer(class_names=class_names)
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
    # print(df.head())
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


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# define prediction function
def predict_probs(text):
    # print(test_x['input_ids'])
    predictions = import_model.predict(x={'input_ids': text})
    # print(predictions)
    x = np.array(list(predictions))
    return np.apply_along_axis(softmax, 1, x)

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
            # model_run = "python {repo_root}\predict.py --root {root} --model {model} --infile {input} --outfile {output} --format {format}".format(
            #     repo_root=repo_root,
            #     root=repo_root,
            #     model = options["model"],
            #     input=in_filename,
            #     output="output.csv",
            #     format = format
            # )
            # print(model_run)
            # os.system(model_run)
            input_csv_path = repo_root+"/"+in_filename
            output_csv_path = repo_root+"/output.csv"
            preds, probs = predict_on_csv(input_csv_path)
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

def predict_on_csv(csv_path):
    data = pd.read_csv(csv_path)
    test_x = tokenizer(
        text=data['reviews'].to_list(),
        add_special_tokens=True,
        max_length=29,
        truncation=True,
        padding='max_length', 
        return_tensors='tf',
        return_token_type_ids = False,
        return_attention_mask = False,
        verbose = True)

    # Run evaluation
    probs = import_model.predict(x={'input_ids': test_x['input_ids']})
    preds = np.argmax(probs, axis=1) + 1
    return preds, probs

def lime_exp(lime, input, n_samples):
    if lime:
        with st.spinner('Calculating...'):
            text = tokenizer(
                text=input,
                add_special_tokens=True,
                max_length=29,
                truncation=True,
                padding = 'max_length', 
                return_tensors='tf',
                return_token_type_ids = False,
                return_attention_mask = False,
                verbose = True)
            # exp = explainer(method,
            #                 path_to_file=METHODS[method]['file'],
            #                 text=text,
            #                 lowercase=METHODS[method]['lowercase'],
            #                 num_samples=int(n_samples))
            # text = 'It was so awesome that it lasted only 15 minutes Hahaha'
            # explain instance with LIME
            exp = explainer.explain_instance(text['input_ids'], predict_probs, num_samples=int(n_samples))
            # exp.show_in_notebook(text=input_text)
            # Display explainer HTML object
            components.html(exp.as_html(), height=800)

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
        options=["BERT"]
    )]
    
    submitted = st.button("Submit")
    submit_clicked(submitted, user_input, df, options, show_final, show_prob)
    
    # title_text = 'LIME Explainer Dashboard'
    # subheader_text = '''1: Strongly Negative &nbsp 2: Weakly Negative &nbsp  3: Neutral &nbsp  4: Weakly Positive &nbsp  5: Strongly Positive'''
    st.title('LIME Explainer Dashboard')
    st.subheader('''1: Strongly Negative &nbsp 2: Weakly Negative &nbsp  3: Neutral &nbsp  4: Weakly Positive &nbsp  5: Strongly Positive''')

    # st.markdown(f"<h2 style='text-align: center;'><b>{title_text}</b></h2>", unsafe_allow_html=True)
    # st.markdown(f"<h5 style='text-align: center;'>{subheader_text}</h5>", unsafe_allow_html=True)
    input_text = st.text_input('Enter your text:', "")
    n_samples = st.text_input('Number of samples to generate for LIME explainer: (For really long input text, go up to 5000)', value=1000)
    # method_list = tuple(label for label, val in METHODS.items())
    # method = st.selectbox(
    #     'Choose classifier:',
    #     method_list,
    #     index=4,
    #     format_func=format_dropdown_labels,
    # )
    lime = st.button("Explain Results")
    lime_exp(lime, input_text, n_samples)
    
        
    st.write('_Developed with ❤️ by [Shreya](https://laddhashreya2000.github.io) & [Hitul]()_')

if __name__ == "__main__":
    main()

def format_dropdown_labels(val):
#     return METHODS[val]['name']
    pass

