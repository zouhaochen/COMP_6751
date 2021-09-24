from nltk.tokenize import word_tokenize
from nltk.corpus import reuters

text = " Welcome readers. I hope you find it interesting. Please do reply."
print(word_tokenize(text))

files = reuters.fileids()
training267 = reuters.words(['training/267'])

print(training267)


