import chardet
def find_encoding(fname):
    f_file = open(fname,'rb').read()
    result = chardet.detect(f_file)
    charenc = result['encoding']
    return charenc
filename = 'Qdata\index.txt'
my_encoding = find_encoding(filename)
with open(filename,'r',encoding = my_encoding) as f:
    lines = f.readlines()
documents = []

def preprocess(document_text):
    terms = [term.lower() for term in document_text.strip().split()[1:]]
    return terms
vocab={}   
for index,line in enumerate(lines):
    tokens = preprocess(line)
    documents.append(preprocess(line))
    for token in tokens:
        if token not in vocab:
            vocab[token] = 1
        else:
            vocab[token] += 1

#reverse sort the vocab by the values
vocab  = dict(sorted(vocab.items(),key=lambda item: item[1],reverse = True))
print('Number of documents: ',len(documents))
print('size of vocab: ',len(vocab))
print('sample document: ',documents[0])
print(vocab)
#save the vocab in a text file
with open('vocab.txt','w') as f:
    for key in vocab.keys():
        f.write("%s\n" % key)
with open('idf-values.txt','w') as f:
    for key in vocab.keys():
        f.write("%s\n" % vocab[key])
with open('documents.txt', 'w') as f:
    for document in documents:
        f.write("%s\n" % ' '.join(document))
inverted_index = {}
for index,document in enumerate(documents):
    for token in document:
        if token not in inverted_index:
            inverted_index[token] = [index]
        else:
            inverted_index[token].append(index)
with open('tf-idf/inverted-index.txt','w') as f:
    for key in inverted_index.keys():
        f.write("%s\n" %key)
        f.write("%s\n"% ' '.join(str(doc_id) for doc_id in inverted_index[key]))


