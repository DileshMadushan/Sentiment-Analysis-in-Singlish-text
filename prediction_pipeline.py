import numpy as np
import pandas as pd
import re
import string
import pickle

from nltk.stem import PorterStemmer
ps = PorterStemmer()

with open('static/model/model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('static/model/corpora/stopwords/english', 'r') as file:
    sw = file.read().splitlines()

with open('static/model/vocabulary.txt', 'r', encoding='utf-8') as f:
    tokens = [line.strip() for line in f]

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def preprocessing(text):
    data = pd.DataFrame([text], columns=['Review (Singlish)'])
    # Convert text to lowercase
    data["Review (Singlish)"] = data["Review (Singlish)"].apply(lambda x: " ".join(x.lower() for x in x.split()))
    # Remove URLs
    data["Review (Singlish)"] = data['Review (Singlish)'].apply(lambda x: " ".join(re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE) for x in x.split()))
    # Remove punctuations
    data["Review (Singlish)"] = data["Review (Singlish)"].apply(remove_punctuations)
    # Remove numbers
    data["Review (Singlish)"] = data['Review (Singlish)'].str.replace(r'\d+', '', regex=True)  # Use raw string
    # Remove stopwords
    data["Review (Singlish)"] = data["Review (Singlish)"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
    # Apply stemming
    data["Review (Singlish)"] = data["Review (Singlish)"].apply(lambda x: " ".join(ps.stem(x) for x in x.split()))
    return data["Review (Singlish)"]

def vectorizer(ds):
    vectorized_lst = []
    for sentence in ds:
        sentence_lst = np.zeros(len(tokens))
        for i in range(len(tokens)):
            if tokens[i] in sentence.split():
                sentence_lst[i] = 1  
        vectorized_lst.append(sentence_lst)
    vectorized_lst_new = np.asarray(vectorized_lst, dtype=np.float32)
    return vectorized_lst_new

def get_prediction(vectorized_text):
    prediction = model.predict(vectorized_text)
    if prediction == 1:
        return 'negative'
    else:
        return 'positive'