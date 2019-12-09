import string
import re
import os
f = open('./test/Bangladesh.txt', 'r')
s = f.read()
corpus_dir = os.path.join(os.getcwd(), "corpus")

out_dir = os.path.join(os.getcwd(), "dataset", "corpus")
if not (os.path.exists(out_dir)):
    os.makedirs(out_dir)          # Create corpus subdirectory in dataset directory

for filename in os.listdir(corpus_dir):
    filepath = corpus_dir + "/" + filename
    f = open(filepath, 'r')
    s = f.read()
    filtered = re.sub(r'\\\\[x][a-zA-Z0-9][a-zA-Z0-9]',' ', s)
    out_filepath = out_dir + "/" + filename
    out_filepath_handle = open(out_filepath, "w")
    out_filepath_handle.write(filtered)
out_filepath_handle.close()


