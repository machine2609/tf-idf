import math
import chardet
from flask import Flask,jsonify
from flask import Flask, render_template, request, jsonify
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
def find_encoding(fname):
    f_file = open(fname,'rb').read()
    result = chardet.detect(f_file)
    charenc = result['encoding']
    return charenc
def load_vocab():
    vocab = {}
    with open('vocab.txt', "r",encoding=find_encoding("vocab.txt")) as f:
        vocab_terms = f.readlines()
    with open('idf-values.txt', "r",encoding=find_encoding("idf-values.txt")) as f:
        idf_values = f.readlines()

    for (term, idf_value) in zip(vocab_terms, idf_values):
        vocab[term.rstrip()] = int(idf_value.rstrip())

    return vocab


def load_document():
    with open("documents.txt", "r",encoding=find_encoding("documents.txt")) as f:
        documents = f.readlines()

    # print('Number of documents: ', len(documents))
    # print('Sample document: ', documents[0])
    return documents
def load_inverted_index():
    inverted_index = {}
    with open('inverted-index.txt', 'r',encoding=find_encoding('inverted-index.txt')) as f:
        inverted_index_terms = f.readlines()

    for row_num in range(0, len(inverted_index_terms), 2):
        term = inverted_index_terms[row_num].strip()
        documents = inverted_index_terms[row_num+1].strip().split()
        inverted_index[term] = documents

    # print('Size of inverted index: ', len(inverted_index))
    return inverted_index

def load_link_of_qs():
    with open("Qindex.txt", "r",encoding=find_encoding("Qindex.txt")) as f:
        links = f.readlines()
    return links
vocab = load_vocab()
documents = load_document()
inverted_index = load_inverted_index()
Qlink = load_link_of_qs()

def get_tf_dictionary(term):
    tf_values = {}
    if term in inverted_index:
        for document in inverted_index[term]:
            if document not in tf_values:
                tf_values[document] = 1
            else:
                tf_values[document] += 1
                
    for document in tf_values:
        try:
            tf_values[document] /= len(documents[int(document)])
        except (ZeroDivisionError, ValueError, IndexError) as e:
            print(e)
            print(document)
    return tf_values

def get_idf_value(term):
    return math.log((1+len(documents))/(1+vocab[term]))
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def calculate_sorted_order_of_documents(query_terms):
    lemmatizer = WordNetLemmatizer()                   # creating a lemmatizer object
    stemmer = PorterStemmer()                           # creating a stemmer object
    stop_words = set(stopwords.words('english'))       # retrieving English stop words
    
    # Lemmatize, stem, and remove stop words from query terms
    lemmatized_terms = [lemmatizer.lemmatize(term) for term in query_terms if term.lower() not in stop_words]
    stemmed_terms = [stemmer.stem(term) for term in lemmatized_terms]
    processed_query_terms = [term.lower().strip() for term in stemmed_terms]

    potential_documents = {}
    ans = []
    for term in processed_query_terms:
        if term not in vocab:
            continue
        tf_values_by_document = get_tf_dictionary(term)
        idf_value = get_idf_value(term)
        for document in tf_values_by_document:
            if document not in potential_documents:
                potential_documents[document] = tf_values_by_document[document] * idf_value
            else:
                potential_documents[document] += tf_values_by_document[document] * idf_value

        for document in potential_documents:
            potential_documents[document] /= len(processed_query_terms)

        potential_documents = dict(sorted(potential_documents.items(), key=lambda item: item[1], reverse=True))

        if len(potential_documents) == 0:
            print("No Matching Document Found. Please search with more relevant terms.")

        for doc_index in potential_documents:
            ans.append({"Question Link": Qlink[int(doc_index)][:-2], "Score": potential_documents[doc_index]})
    return ans


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
# query_string = input('Enter your query: ')
# query_terms = [term.lower() for term in query_string.strip().split()]
class SearchForm(FlaskForm):
    search = StringField('Enter your search term')
    submit = SubmitField('Search')


@app.route("/<query>")
def return_links(query):
    q_terms = [term.lower() for term in query.strip().split()]
    return jsonify(calculate_sorted_order_of_documents(q_terms)[:20:])


@app.route("/", methods=['GET', 'POST'])
def home():
    form = SearchForm()
    results = []
    if form.validate_on_submit():
        query = form.search.data
        q_terms = [term.lower() for term in query.strip().split()]
        results = calculate_sorted_order_of_documents(q_terms)[:20:]
    return render_template('index.html', form=form, results=results)
if __name__ == "__main__":
    app.run(debug = True)
