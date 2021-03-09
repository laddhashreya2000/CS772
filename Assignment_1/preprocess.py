
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