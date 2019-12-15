
############# Point 1

# importing libraries 
import os, os.path 
  
# using the given path 
path = os.path.expanduser('~/nltk_data') 
  
# checking 
if not os.path.exists(path): 
    os.mkdir(path) 
	  
print ("Does path exists : ", os.path.exists(path)) 
  
# import nltk.data 
# print ("\nDoes path exists in nltk : ",  
#        path in nltk.data.path) 


############# Point 2

import os
import re
import nltk
import codecs
import string
import subprocess
import unicodedata
## Set important paths
DOCS_PDF = os.path.join(os.getcwd(), "pdf")

def get_pdf_docs(path=DOCS_PDF):
	"""
	Corpus
	Returns a filtered list of paths to PDF files
	"""
	print(path)
	for name in os.listdir(path):
		if name.endswith('.pdf'):
			yield os.path.join(path, name)
#total number documents
print(len(list(get_pdf_docs())))

## Create a path to extract the corpus.
CORPUS = os.path.join(os.getcwd(), "corpus")

def extract_pdf_corpus(DOCS_PDF=DOCS_PDF, corpusval=CORPUS):
	"""
	Extracts a text corpus from the PDF documents 
	"""
	# Create corpus directory if it doesn't exist.
	if not os.path.exists(corpusval):
		os.mkdir(corpusval)
	# For each PDF path, use pdf2txt to extract the text file.
	for path in get_pdf_docs(DOCS_PDF):
		# Call the subprocess command (must be on your path)
		tp=os.getcwd()+'\\pdf2txt.py'
		document = subprocess.check_output(
			['python',tp, path]
		)

		# document=str(document)

		# print(str(document))
		# print(type(document))


		# Write the document out to the  directory corpus
		filen = os.path.splitext(os.path.basename(path))[0] + ".txt"
		outp = os.path.join(corpusval, filen)
		# print(document)
		############# Point 3

		document=document.decode("utf-8") 
		document=document.replace('\n',' ')
		document=document.replace('\r',' ')
		document=document.replace(',',' ')
		document=document.replace('-','')
		document=document.replace(':',' ')
		
		document=str(document)

		############# Point 4

		# document=document.encode("utf-8") 
		document=str(document)

		with codecs.open(outp, 'w', encoding='utf-8') as f:
			f.write(document)
# Run the extraction
extract_pdf_corpus()
# Create an NLTK corpus reader to access text data on disk.                
kddcorpus = nltk.corpus.PlaintextCorpusReader(CORPUS, '.*\.txt')
############# Point 5,6,7,8

wordsvals = nltk.FreqDist(kddcorpus.words())
count = sum(wordsvals.values())
vocab = len(wordsvals)
print('----------------------------------------------------------------------')
print("Corpus contains a vocabulary of {} and a word count of {}.".format(
	count, vocab
))

# print(wordsvals.values)
# print(wordsvals.items)
print('----------------------------------------------------------------------')
print(wordsvals.hapaxes)

print('----------------------------------------------------------------------')
print(wordsvals.most_common)
