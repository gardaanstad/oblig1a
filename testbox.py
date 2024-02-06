import json
import regex
import nltk
import nltk.data
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from helpers_1a import scatter_plot

# NLTK-ressurser vi skal ha tilgjengelig i denne obligen
resources = {"punkt": "tokenizers/punkt"}

for name, path in resources.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(name)

def prepare_data(documents, split): # Oppgave 1 a    
    # Din kode her
    def meets_requirements(document, split):
        if not document.get("metadata").get("split") == split:
            return False
        if document.get("metadata").get("category") not in ["games", "restaurants", "literature"]:
            return False
        return True
    
    data = [d.get("text")
            for d in documents 
            if meets_requirements(d, split)]
    
    labels = [d.get("metadata").get("category")
              for d in documents
              if meets_requirements(d, split)]
    
    data = list(data)
    labels = list(labels)
    
    assert len(data) == len(labels)
    return data, labels

# Prekode: Her laster vi inn dataene, etter at du har skrevet ferdig prepare_data()
datakilde = "norec_excerpts.json"
with open (datakilde, encoding = "utf-8") as rf:
    norecdata = json.load(rf)

# Treningsdata
train_data, train_labels = prepare_data(norecdata, "train")

# Valideringsdata
# dev_data, dev_labels = prepare_data(norecdata, "dev")

# Testdata
# test_data, test_labels = prepare_data(norecdata, "test")

def tokenize(text): #Oppgave 1 b
    """Tar inn en streng med tekst og returnerer en liste med tokens."""
    # Å splitte på mellomrom er fattigmanns tokenisering. Endre til noe bedre!
    tokenized = word_tokenize(text)

    return tokenized

all_tokens = []
for text in train_data:
    tokenized = tokenize(text)
    for token in tokenized:
        all_tokens.append(token)

amount_of_tokens = len(all_tokens)
unique_tokens = set(all_tokens)
amount_of_unique_tokens = len(unique_tokens)

print(all_tokens[:3])
print(amount_of_tokens)
print(amount_of_unique_tokens)