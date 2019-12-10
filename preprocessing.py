import string
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from string import punctuation

nltk.download('wordnet')

"""
1. Convert text to lower case:
"""
def to_lower(text):
    return ' '.join([w.lower() for w in word_tokenize(text)])
"""
2. remove numbers
"""
def remove_digit(text):
    return ''.join(c for c in text if not c.isdigit())

"""
3. Remove punctuation
"""

def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)

"""
4. Lemmatize
"""
stopword = stopwords.words('english')
wordnet_lemmatizer = WordNetLemmatizer()
def lemmatizing(text):
    word_tokens = word_tokenize(text)
    lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in word_tokens]
    return ' '.join(s for s in lemmatized_word)

"""
5. Stemming
"""
porter_stemmer = PorterStemmer()
def stemming(text):
    word_tokens = nltk.word_tokenize(text)
    stemmed_word = [porter_stemmer.stem(word) for word in word_tokens]
    return ' '.join(s for s in stemmed_word)

"""
6. Remove short words
"""
def remove_short_words(text):
    ret_text = ''
    word_tokens = nltk.word_tokenize(text)
    for token in word_tokens:
        if len(token) > 2:
            ret_text += token + ' '
    return ret_text


corpus_dir = os.path.join(os.getcwd(), "corpus")

full_out_dir = os.path.join(os.getcwd(), "dataset")
stemming_out_dir = os.path.join(os.getcwd(), "dataset_stemming")
lemmatizing_out_dir = os.path.join(os.getcwd(), "dataset_lemmatizing")
if not (os.path.exists(full_out_dir)):
    os.makedirs(full_out_dir)          
if not (os.path.exists(stemming_out_dir)):
    os.makedirs(stemming_out_dir)          
if not (os.path.exists(lemmatizing_out_dir)):
    os.makedirs(lemmatizing_out_dir)          


full_out_dir = os.path.join(os.getcwd(), "dataset", "corpus")
stemming_out_dir = os.path.join(os.getcwd(), "dataset_stemming", "corpus")
lemmatizing_out_dir = os.path.join(os.getcwd(), "dataset_lemmatizing", "corpus")
if not (os.path.exists(full_out_dir)):
    os.makedirs(full_out_dir)          
if not (os.path.exists(stemming_out_dir)):
    os.makedirs(stemming_out_dir)          
if not (os.path.exists(lemmatizing_out_dir)):
    os.makedirs(lemmatizing_out_dir)          

for filename in os.listdir(corpus_dir):
    filepath = corpus_dir + "/" + filename
    f = open(filepath, 'r')
    s = f.read()
    filtered = re.sub(r'\\\\[x][a-zA-Z0-9][a-zA-Z0-9]',' ', s)
    # text to lower    
    filtered = to_lower(filtered)
    # remove numbers
    filtered = remove_digit(filtered)
    # remove puntuation
    filtered = strip_punctuation(filtered)
    # lemmatizing
    lemmatizing_filtered = lemmatizing(filtered)
    # remove short words
    lemmatizing_filtered = remove_short_words(lemmatizing_filtered)
    out_filepath = lemmatizing_out_dir + "/" + filename
    out_filepath_handle = open(out_filepath, "w")
    out_filepath_handle.write(lemmatizing_filtered)
    out_filepath_handle.close()

    # stemming
    stemming_filtered = stemming(filtered)
    # remove short words
    stemming_filtered = remove_short_words(stemming_filtered)
    filtered = stemming(lemmatizing_filtered)
    # remove short words
    filtered = remove_short_words(filtered)    
    out_filepath = stemming_out_dir + "/" + filename
    out_filepath_handle = open(out_filepath, "w")
    out_filepath_handle.write(stemming_filtered)
    out_filepath_handle.close()

    out_filepath = full_out_dir + "/" + filename
    out_filepath_handle = open(out_filepath, "w")
    out_filepath_handle.write(filtered)
    out_filepath_handle.close()


