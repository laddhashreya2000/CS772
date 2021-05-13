# A Unified MRC Framework for Named Entity Recognition

## Problem Statement
A unified framework capable of handling both flat and nested NER tasks formulated with an MRC approach.
Reference paper - https://arxiv.org/abs/1910.11476
Reference repository -  https://github.com/manliu1225/mrc-for-flat-nested-ner

## Usage

### Model training
```console
> #Download the zip folder and unzip it
> cd mrc-for-flat-nested-ner
```
Download the pretrained base model `BERT-base Uncased 12-layer, 768-hidden, 12-heads, 110M parameters` from [here](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) and unzip it in this root folder. This will give us the tensorflow checkpoints. They have to be converted into pytorch model with the script `convert_tf_checkpoint_to_pytorch.py` and saved as `pytorch_model.bin` in `uncased_L-12_H-768_A-12` folder.

### Dataset
We used the dataset ACE 2004 Multilingual corpus. It had to be modified to add query text corresponding to each tag. The modified dataset can be downloaded from here https://drive.google.com/file/d/1KHRSTL_jn5PxQqz4prQ1No2E2wWcuxOd/view?usp=sharing. Store the folder en_ace04 in root folder.

```
> pip install -r requirements.txt
> python3 run.py
```
Note: Change the number of GPU's argument(n_gpu) in the arg_parse function accroding to the number of gpus in use. The trained model will get saved in the output folder in the root directory. Move this saved model to root directory and change the name in `evaluate.py`.

### Model prediction

Make a input data file similar to `en_ace04/mrc-ner.test` which contains 7 queries related to 7 labels for one context. Provide the path to dataset in `evaluate.py`. Run `python3 evaluate.py` will print in terminal the actual and predicted start and the end positions for each entity label. 

---
Presentation slides link - https://drive.google.com/file/d/1CjK6_C9FDcvrgJyREwSXyRiwZw4om74_/view?usp=sharing