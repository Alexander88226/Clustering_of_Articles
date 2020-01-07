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
import unicodedata

new_stop_words = ["a","about","above","after","again","against","ain","all","am","an","and","any","are","aren","aren't","as","at","be","because",\
    "been","before","being","below","between","both","but","by","can","couldn","couldn't","d","did","didn","didn't","do","does","doesn","doesn't",\
    "doing","don","don't","down","during","each","few","for","from","further","had","hadn","hadn't","has","hasn","hasn't","have","haven","haven't",\
    "having","he","her","here","hers","herself","him","himself","his","how","i","if","in","into","is","isn","isn't","it","it's","its","itself",\
    "just","ll","m","ma","me","mightn","mightn't","more","most","mustn","mustn't","my","myself","needn","needn't","no","nor","not","now","o","of",\
    "off","on","once","only","or","other","our","ours","ourselves","out","over","own","re","s","same","shan","shan't","she","she's","should",\
    "should've","shouldn","shouldn't","so","some","such","t","than","that","that'll","the","their","theirs","them","themselves","then","there",\
    "these","they","this","those","through","to","too","under","until","up","ve","very","was","wasn","wasn't","we","were","weren","weren't","what",\
    "when","where","which","while","who","whom","why","will","with","won","won't","wouldn","wouldn't","y","you","you'd","you'll","you're","you've",\
    "your","yours","yourself","yourselves","could","he'd","he'll","he's","here's","how's","i'd","i'll","i'm","i've","let's","ought","she'd","she'll",\
    "that's","there's","they'd","they'll","they're","they've","we'd","we'll","we're","we've","what's","when's","where's","who's","why's","would",\
    "able","abst","accordance","according","accordingly","across","act","actually","added","adj","affected","affecting","affects","afterwards","ah",\
    "almost","alone","along","already","also","although","always","among","amongst","announce","another","anybody","anyhow","anymore","anyone",\
    "anything","anyway","anyways","anywhere","apparently","approximately","arent","arise","around","aside","ask","asking","auth","available","away",\
    "awfully","b","back","became","become","becomes","becoming","beforehand","begin","beginning","beginnings","begins","behind","believe","beside",\
    "besides","beyond","biol","brief","briefly","c","ca","came","cannot","can't","cause","causes","certain","certainly","co","com","come","comes",\
    "contain","containing","contains","couldnt","date","different","done","downwards","due","e","ed","edu","effect","eg","eight","eighty","either",\
    "else","elsewhere","end","ending","enough","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex",\
    "except","f","far","ff","fifth","first","five","fix","followed","following","follows","former","formerly","forth","found","four","furthermore",\
    "g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","happens","hardly","hed","hence","hereafter",\
    "hereby","herein","heres","hereupon","hes","hi","hid","hither","home","howbeit","however","hundred","id","ie","im","immediate","immediately","importance",\
    "important","inc","indeed","index","information","instead","invention","inward","itd","it'll","j","k","keep","keeps","kept","kg","km","know","known",\
    "knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll",\
    "look","looking","looks","ltd","made","mainly","make","makes","many","may","maybe","mean","means","meantime","meanwhile","merely","mg","might",\
    "million","miss","ml","moreover","mostly","mr","mrs","much","mug","must","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary",\
    "need","needs","neither","never","nevertheless","new","next","nine","ninety","nobody","non","none","nonetheless","noone","normally","nos","noted",\
    "nothing","nowhere","obtain","obtained","obviously","often","oh","ok","okay","old","omitted","one","ones","onto","ord","others","otherwise","outside",\
    "overall","owing","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly",\
    "potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv",\
    "r","ran","rather","rd","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research",\
    "respectively","resulted","resulting","results","right","run","said","saw","say","saying","says","sec","section","see","seeing","seem","seemed",\
    "seeming","seems","seen","self","selves","sent","seven","several","shall","shed","shes","show","showed","shown","showns","shows","significant",\
    "significantly","similar","similarly","since","six","slightly","somebody","somehow","someone","somethan","something","sometime","sometimes",\
    "somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","still","stop","strongly","sub","substantially",\
    "successfully","sufficiently","suggest","sup","sure","take","taken","taking","tell","tends","th","thank","thanks","thanx","thats","that've",\
    "thence","thereafter","thereby","thered","therefore","therein","there'll","thereof","therere","theres","thereto","thereupon","there've","theyd",\
    "theyre","think","thou","though","thoughh","thousand","throug","throughout","thru","thus","til","tip","together","took","toward","towards",\
    "tried","tries","truly","try","trying","ts","twice","two","u","un","unfortunately","unless","unlike","unlikely","unto","upon","ups","us","use",\
    "used","useful","usefully","usefulness","uses","using","usually","v","value","various","'ve","via","viz","vol","vols","vs","w","want","wants",\
    "wasnt","way","wed","welcome","went","werent","whatever","what'll","whats","whence","whenever","whereafter","whereas","whereby","wherein",\
    "wheres","whereupon","wherever","whether","whim","whither","whod","whoever","whole","who'll","whomever","whos","whose","widely","willing",\
    "wish","within","without","wont","words","world","wouldnt","www","x","yes","yet","youd","youre","z","zero","a's","ain't","allow","allows",\
    "apart","appear","appreciate","appropriate","associated","best","better","c'mon","c's","cant","changes","clearly","concerning","consequently",\
    "consider","considering","corresponding","course","currently","definitely","described","despite","entirely","exactly","example","going","greetings",\
    "hello","help","hopefully","ignored","inasmuch","indicate","indicated","indicates","inner","insofar","it'd","keep","keeps","novel","presumably",\
    "reasonably","second","secondly","sensible","serious","seriously","sure","t's","third","thorough","thoroughly","three","well","wonder"]

stop_words = list(set(stopwords.words('english')))
stop_words = set(stop_words + new_stop_words)
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

import spacy
# Initialize spacy 'en' model, keeping only tagger component needed for lemmatization
nlp = spacy.load('en', disable=['parser', 'ner'])

def lemmatizing_spacy(text):
    # Parse the sentence using the loaded 'en' model object `nlp`
    doc = nlp(text)

    # Extract the lemma for each token and join
    return " ".join([token.lemma_ for token in doc])

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
"""
9. Replace 
"""
def replaceUnicode(text):
    ret_text = ''
    word_tokens = nltk.word_tokenize(text)
    for token in word_tokens:
        ret_text += unicodedata.normalize("NFKD", token) + ' '
    return ret_text

corpus_dir = os.path.join(os.getcwd(), "corpus")

lemmatizing_out_dir = os.path.join(os.getcwd(), "pre_dataset_lemmatizing")
if not (os.path.exists(lemmatizing_out_dir)):
    os.makedirs(lemmatizing_out_dir)          


lemmatizing_out_dir = os.path.join(os.getcwd(), "pre_dataset_lemmatizing", "corpus")
if not (os.path.exists(lemmatizing_out_dir)):
    os.makedirs(lemmatizing_out_dir)          

# stemming_out_dir = os.path.join(os.getcwd(), "pre_dataset_stemming")
# if not (os.path.exists(stemming_out_dir)):
#     os.makedirs(stemming_out_dir)          


# stemming_out_dir = os.path.join(os.getcwd(), "pre_dataset_stemming", "corpus")
# if not (os.path.exists(stemming_out_dir)):
#     os.makedirs(stemming_out_dir)          

# full_out_dir = os.path.join(os.getcwd(), "pre_dataset_full")
# if not (os.path.exists(full_out_dir)):
#     os.makedirs(full_out_dir)          


# full_out_dir = os.path.join(os.getcwd(), "pre_dataset_full", "corpus")
# if not (os.path.exists(full_out_dir)):
#     os.makedirs(full_out_dir)          

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
    # lemmatizing_filtered = lemmatizing_spacy(lemmatizing_filtered)
    # # stemming
    # stemming_filtered = stemming(filtered)
    # full_filtered = stemming(lemmatizing_filtered)
    # remove short words
    lemmatizing_filtered = remove_short_words(lemmatizing_filtered)
    # stemming_filtered = remove_short_words(stemming_filtered)
    # full_filtered = remove_short_words(full_filtered)

    # replcace U+FB to english words
    lemmatizing_filtered = replaceUnicode(lemmatizing_filtered)
    # write the filtered text into file
    out_filepath = lemmatizing_out_dir + "/" + filename
    out_filepath_handle = open(out_filepath, "w", encoding='utf-8')
    out_filepath_handle.write(lemmatizing_filtered)
    out_filepath_handle.close()

    # out_filepath = stemming_out_dir + "/" + filename
    # out_filepath_handle = open(out_filepath, "w", encoding='utf-8')
    # out_filepath_handle.write(stemming_filtered)
    # out_filepath_handle.close()

    # out_filepath = full_out_dir + "/" + filename
    # out_filepath_handle = open(out_filepath, "w", encoding='utf-8')
    # out_filepath_handle.write(full_filtered)
    # out_filepath_handle.close()

# get 50 top frequent words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import datasets

def remove_top_freq_stopwords(text, stopword_list):
    tokens = word_tokenize(text)
    text = ' '.join(t for t in tokens if not t in stopword_list)
    tokens = word_tokenize(text)
    new_text = ''
    for token in tokens:
        flag = False
        for stopword in stopword_list:
            if stopword in token:
                flag = True
        if not flag:
            new_text += token + ' '
    return new_text

def get_top_n_words_n_que(corpus, n=None):
    
    vec = CountVectorizer(stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq_que = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq_que, key = lambda x: x[1], reverse=True)
    return words_freq[:n], words_freq_que[:n]

def remove_50_top_words_from_corpus(dataset_dir_str):
    dataSetDir2 = os.path.join(os.getcwd(), "pre_" + dataset_dir_str)
    all_data = datasets.load_files(dataSetDir2, description=None, load_content=True, encoding='utf-8', shuffle=False)
    top_n_words, que_n_words = get_top_n_words_n_que(all_data.data, 50);
    #-------------------------------------------

    # remove 50 top frequent words from corpus

    corpus_out_dir = os.path.join(os.getcwd(), dataset_dir_str, "corpus")
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

remove_50_top_words_from_corpus("dataset_lemmatizing")
# remove_50_top_words_from_corpus("dataset_stemming")
# remove_50_top_words_from_corpus("dataset_full")