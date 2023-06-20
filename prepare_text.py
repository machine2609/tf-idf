import os
import chardet
import re
folder_path = "Qdata"  # Replace with the path to your top-level folder
stop_line = "Example 1:"  # Replace with the line after which you want to stop reading


def find_encoding(fname):
    f_file = open(fname, 'rb').read()
    result = chardet.detect(f_file)
    charenc = result['encoding']
    return charenc


def preprocess(document_text):
    terms = [term.lower() for term in document_text.strip().split()]
    return terms


vocab = {}
documents = []
all_lines = []
for i in range(1, 1735):
    file_path = os.path.join(folder_path, "{}/{}.txt".format(i, i))

    doc = ""
    with open(file_path, "r", encoding= 'utf-8', errors = "ignore") as f:
        lines = f.readlines()
    with open('tf-idf\index.txt', "r", encoding= 'utf-8', errors = "ignore") as f:
        lines_heading = f.readlines()
    for line in lines:
        if stop_line in line:
            break
        else:
            doc += line
    doc+=lines_heading[i-1][1:]
    all_lines.append(doc)

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def preprocess(text):
    text = re.sub(r'[^a-zA-Z0-9\s-]', '', text)      # removing non-alphanumeric chars
    tokens = nltk.word_tokenize(text)                 # tokenizing the text
    lemmatizer = WordNetLemmatizer()                   # creating a lemmatizer object
    stemmer = PorterStemmer()                           # creating a stemmer object
    stop_words = set(stopwords.words('english'))       # retrieving English stop words

    # lemmatizing and stemming each token, and removing stop words
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.lower() not in stop_words]
    stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]

    # converting tokens to lowercase and removing leading/trailing whitespaces
    normalized_tokens = [token.lower().strip() for token in stemmed_tokens]

    return normalized_tokens


vocab = {}
documents = []

for(index,line) in enumerate(all_lines):
    tokens = preprocess(line)
    documents.append(tokens)
    tokens = set(tokens)
    for token in tokens:
        if token not in vocab:
            vocab[token] = 1
        else:
            vocab[token] += 1

vocab = dict( sorted(vocab.items(), key = lambda item : item[1], reverse = True) )
print("No of documents : ", len(documents))
print("Size of vocab : ", len(vocab))
print("Sample document: ", documents[0])
with open('tf-idf/vocab.txt', 'w', encoding='utf-8') as f:
    for key in vocab.keys():
        f.write("%s\n" % key)

with open('tf-idf/idf-values.txt', 'w', encoding='utf-8') as f:
    for key in vocab.keys():
        f.write("%s\n" % vocab[key])

with open('tf-idf/documents.txt', 'w', encoding='utf-8') as f:
    for document in documents:
        f.write("%s\n" % document)

inverted_index = {}
for index, document in enumerate(documents):
    for token in document:
        if token not in inverted_index:
            inverted_index[token] = [index]
        else:
            inverted_index[token].append(index)

with open('tf-idf/inverted-index.txt', 'w', encoding='utf-8') as f:
    for key in inverted_index.keys():
        f.write("%s\n" % key)
        f.write("%s\n" % ' '.join(str(doc_id) for doc_id in inverted_index[key]))