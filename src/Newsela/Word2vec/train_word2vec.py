import gensim
from gensim.models import Word2Vec 
from io import open
import string
import io
import unicodedata

train_src = open('../../../data-simplification/newsela/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.train.src', encoding='utf-8').read().split('\n')
train_dst = open('../../../data-simplification/newsela/V0V4_V1V4_V2V4_V3V4_V0V3_V0V2_V1V3.aner.ori.train.dst', encoding='utf-8').read().split('\n')

#k = open('sentences.txt', 'a')

def unicodeToAscii(s):
	return ''.join(
	c for c in unicodedata.normalize('NFD', s)
	if unicodedata.category(c) != 'Mn'
	)

l = []
for i in range(len(train_src)-1):
	a = unicodeToAscii(train_src[i].lower()).split(' ')
	l.append(a)
	'''k.write(unicodeToAscii(train_src[i].lower()))
	k.write('\n')
	k.write(unicodeToAscii(train_dst[i].lower()))
	k.write('\n')'''
	l.append(unicodeToAscii(train_dst[i].lower()).split(' '))

'''l = []
lyr = open('final_lyrics.txt')
b = lyr.readlines()
for i in range(len(b)):
	a = b[i].split(' ')
	p = len(a)
	#print(a[p-1][-1])
	if a[p-1][-1:] == '\n':
		a[p-1] = a[p-1][:-1]
	l.append(a)'''

#docs = [["cat", "say", "meow"], ["dog", "say", "woof"]]     
model = Word2Vec(l, size = 300, window = 5, min_count = 1, workers = 4)
print(model.similar_by_word('able-bodied', topn=10, restrict_vocab=None))
print(len(model.wv.vocab))
model.save("word2vec.model")
model = Word2Vec.load('word2vec.model')
print(model.similar_by_word('principal', topn=10, restrict_vocab=None))

