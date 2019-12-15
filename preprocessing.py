import string
import re
import os
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from string import punctuation
from langdetect import detect, detect_langs



stop_words = set(stopwords.words('english'))

nltk.download('averaged_perceptron_tagger')

nltk.download('wordnet')

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

"""
1. Convert text to lower case:
"""
def to_lower(text):
    return ' '.join([w.lower() for w in word_tokenize(text)])

"""
2. Remove stopwords
"""
def remove_stopwords(text):
    tokens = word_tokenize(text)
    return ' '.join(t for t in tokens if not t in stop_words)

"""
3. remove numbers
"""
def remove_digit(text):
    return ''.join(c for c in text if not c.isdigit())

"""
4. Remove punctuation
"""

def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation+'–')

"""
5. Lemmatize
"""
stopword = stopwords.words('english')
wordnet_lemmatizer = WordNetLemmatizer()
def lemmatizing(text):
    word_tokens = word_tokenize(text)
    lemmatized_word = [wordnet_lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in word_tokens]
    return ' '.join(s for s in lemmatized_word)

"""
6. Stemming
"""
porter_stemmer = PorterStemmer()
def stemming(text):
    word_tokens = nltk.word_tokenize(text)
    stemmed_word = [porter_stemmer.stem(word) for word in word_tokens]
    return ' '.join(s for s in stemmed_word)

"""
7. Remove short words
"""
def remove_short_words(text):
    ret_text = ''
    word_tokens = nltk.word_tokenize(text)
    for token in word_tokens:
        if len(token) > 2:
            ret_text += token + ' '
    return ret_text
"""
8. Detect language of string
"""
def getLang(text):
    langs = detect_langs(text)
    for lang in langs:
        if lang.lang == 'en' and lang.prob > 0.01:
            return True, langs
    return False, langs
    # word_tokens = nltk.word_tokenize(text)
    # total_count = len(word_tokens)
    # eng_count = 0
    # for token in word_tokens:
        
    #     for lang in langs:
    #         if lang.lang == 'en' and lang 
    # if eng_count / total_count > 0.8:
    #     return True
    # else:
    #     return False



corpus_dir = os.path.join(os.getcwd(), "corpus")

lemmatizing_out_dir = os.path.join(os.getcwd(), "dataset_lemmatizing")
if not (os.path.exists(lemmatizing_out_dir)):
    os.makedirs(lemmatizing_out_dir)          


lemmatizing_out_dir = os.path.join(os.getcwd(), "dataset_lemmatizing", "corpus")
if not (os.path.exists(lemmatizing_out_dir)):
    os.makedirs(lemmatizing_out_dir)          

for filename in os.listdir(corpus_dir):
    filepath = corpus_dir + "/" + filename
    f = open(filepath, 'r', encoding='utf-8')
    s = f.read()
    filtered = re.sub(r'\\[x][a-zA-Z0-9][a-zA-Z0-9]',' ', s)
    # text to lower    
    filtered = to_lower(filtered)    
    # remove numbers
    filtered = remove_digit(filtered)

    # detect language of document
    ret,langs = getLang(filtered)
    if not ret:
        print(filepath + ':')  
        print(getLang(filtered)) 
        print(langs)
        continue

    # remove stopwords
    filtered = remove_stopwords(filtered)
    # remove puntuation
    filtered = strip_punctuation(filtered)
    # lemmatizing
    lemmatizing_filtered = lemmatizing(filtered)
    # remove short words
    lemmatizing_filtered = remove_short_words(lemmatizing_filtered)
    out_filepath = lemmatizing_out_dir + "/" + filename
    out_filepath_handle = open(out_filepath, "w", encoding='utf-8')
    out_filepath_handle.write(lemmatizing_filtered)
    out_filepath_handle.close()

# get 50 top frequent words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import datasets

def remove_top_freq_stopwords(text, stopword_list):
    tokens = word_tokenize(text)
    return ' '.join(t for t in tokens if not t in stopword_list)

def get_top_n_words_n_que(corpus, n=None):
    
    vec = CountVectorizer(stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq_que = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq_que, key = lambda x: x[1], reverse=True)
    return words_freq[:n], words_freq_que[:n]

dataSetDir2 = os.path.join(os.getcwd(), "dataset_lemmatizing")
all_data = datasets.load_files(dataSetDir2, description=None, load_content=True, encoding='utf-8', shuffle=False)
top_n_words, que_n_words = get_top_n_words_n_que(all_data.data, 50);
#-------------------------------------------

# remove 50 top frequent words from corpus

corpus_out_dir = os.path.join(os.getcwd(), "dataset", "corpus")
if not (os.path.exists(corpus_out_dir)):
    os.makedirs(corpus_out_dir)          
stopwords = [word for (word,freq) in top_n_words]
print(stopwords)

for filename in os.listdir(lemmatizing_out_dir):
    filepath = lemmatizing_out_dir + "/" + filename
    f = open(filepath, 'r', encoding='utf-8')
    s = f.read()
    filtered = remove_top_freq_stopwords(s, stopwords)
    out_filepath = corpus_out_dir + "/" + filename
    out_filepath_handle = open(out_filepath, "w", encoding='utf-8')
    out_filepath_handle.write(filtered)
    out_filepath_handle.close()