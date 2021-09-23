from nltk.corpus import reuters

files = reuters.fileids()

words9920 = reuters.words(['training/9920'])[:71]
print(words9920)