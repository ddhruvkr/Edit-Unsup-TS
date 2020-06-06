from model.sari_org import calculate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
#from nltk.translate.meteor_score import single_meteor_score
sf = SmoothingFunction()
from model.FKGL import sentence_fre_final, sentence_fkgl
#from readability.readability import Readability
#from readability.utils_rb import *
import numpy as np
from io import open
import unicodedata
# scp -r mcmc_sampling/readability_py2 d35kumar2@lg-research-2:~/Github/Text_Simplification/mcmc_sampling/
make_file = True # this should be True, if you want to create a new file from the file created by tree_edits_beam.py. The new file will have sentences in each line (which is the format in which the output is given for computing metircs)
# and should be false, if you want to calculate scores for the file with sentences

dataset = 'Wikilarge' #Wikilarge, Newsela
m = 'Wikilarge/output/simplifications_Wikilarge.txt'
n = 'Wikilarge/simple/simplifications_Wikilarge_simple_hs.txt'

print("All evalulations should be done on corpus level")
#n= 'Wikilarge/sbmt.txt'
def unicodeToAscii(s):
	return ''.join(
	c for c in unicodedata.normalize('NFD', s)
	if unicodedata.category(c) != 'Mn')

def convert_to_blue(sent):
	s = []
	for i in sent.split(' '):
		s.append(i)
	#print(s)
	return s


if make_file:
	Mine = open(m, encoding='utf-8').read().split('\n')
	i = 2
	k = 0
	if dataset == 'Wikilarge':
		val = 359
	elif dataset == 'Newsela':
		val = 1077
	with open(n, "a") as file:
		# 1077, 1076, #359, 358
		while k < val:
			if k != val-1:
				#print(i)
				file.write(Mine[i] + '\n')
			else:
				file.write(Mine[i])
			i += 8
			k += 1

elif dataset == 'Wikilarge':

	Complex = open('Wikilarge/turkcorpus/test.8turkers.tok.norm', encoding='utf-8').read().split('\n')
	#DressLs = open('Wikilarge/dress-ls.txt', encoding='utf-8').read().split('\n')
	#Dress = open('Wikilarge/dress.txt', encoding='utf-8').read().split('\n')
	#EncDecA = open('Wikilarge/encdec.txt', encoding='utf-8').read().split('\n')
	#Hybrid = open('Wikilarge/hybrid.txt', encoding='utf-8').read().split('\n')
	#PBMTR = open('Wikilarge/pbmt.txt', encoding='utf-8').read().split('\n')
	#SBMT = open('Wikilarge/sbmt.txt', encoding='utf-8').read().split('\n')
	#Access = open('Wikilarge/access.txt', encoding='utf-8').read().split('\n')
	#Surya = open('Wikilarge/surya2.txt', encoding='utf-8').read().split('\n')
	#S2SAllFA = open('metrics/S2S-All-FA.txt', encoding='utf-8').read().split('\n')
	#EditNTS = open('Wikilarge/wikilarge_editnts.txt', encoding='utf-8').read().split('\n')
	#EntPar = open('metrics/EntPar.txt', encoding='utf-8').read().split('\n')
	Reference = open('Wikilarge/turkcorpus/test.8turkers.tok.simp', encoding='utf-8').read().split('\n')
	Reference0 = open('Wikilarge/turkcorpus/test.8turkers.tok.turk.0', encoding='utf-8').read().split('\n')
	Reference1 = open('Wikilarge/turkcorpus/test.8turkers.tok.turk.1', encoding='utf-8').read().split('\n')
	Reference2 = open('Wikilarge/turkcorpus/test.8turkers.tok.turk.2', encoding='utf-8').read().split('\n')
	Reference3 = open('Wikilarge/turkcorpus/test.8turkers.tok.turk.3', encoding='utf-8').read().split('\n')
	Reference4 = open('Wikilarge/turkcorpus/test.8turkers.tok.turk.4', encoding='utf-8').read().split('\n')
	Reference5 = open('Wikilarge/turkcorpus/test.8turkers.tok.turk.5', encoding='utf-8').read().split('\n')
	Reference6 = open('Wikilarge/turkcorpus/test.8turkers.tok.turk.6', encoding='utf-8').read().split('\n')
	Reference7 = open('Wikilarge/turkcorpus/test.8turkers.tok.turk.7', encoding='utf-8').read().split('\n')
	#Baseline = open('base250.txt', encoding='utf-8').read().split('\n')
	Mine = open(n, encoding='utf-8').read().split('\n')
	#Mine = Reference
	#Mine = Baseline
	'''with open('metrics/Mine/'+n, "a") as file:
		l = len(Reference)
		for i in range(l):
			file.write(unicodeToAscii(EncDecA[i].lower()) + '\n')'''

	sari = 0.0
	keep = 0.0
	delete = 0.0
	add = 0.0
	bleu = 0.0
	f = 0.0
	fr = 0.0
	avg_len = 0
	candidate = []
	reference = []
	reference_sent = ''
	l = len(Reference0)
	for i in range(l):
		avg_len += len(Mine[i].split(' ')) 
		a0 = unicodeToAscii(Reference0[i].lower())
		a1 = unicodeToAscii(Reference1[i].lower())
		a2 = unicodeToAscii(Reference2[i].lower())
		a3 = unicodeToAscii(Reference3[i].lower())
		a4 = unicodeToAscii(Reference4[i].lower())
		a5 = unicodeToAscii(Reference5[i].lower())
		a6 = unicodeToAscii(Reference6[i].lower())
		a7 = unicodeToAscii(Reference7[i].lower())
		#a = Reference[i]
		b = unicodeToAscii(Complex[i].lower())
		#b = Complex[i]
		c = unicodeToAscii(Mine[i].lower())
		#c = Dress[i]
		f += sentence_fkgl(c)
		fr += sentence_fre_final(c)
		candidate.append(convert_to_blue(c))
		reference.append([convert_to_blue(a0),convert_to_blue(a1),convert_to_blue(a2),convert_to_blue(a3),
			convert_to_blue(a4),convert_to_blue(a5),convert_to_blue(a6),convert_to_blue(a7)])
		s, k, d, ad = calculate(b, c, [a0,a1,a2,a3,a4,a5,a6,a7])
		sari += s
		keep += k
		delete += d
		add += ad
		'''print([convert_to_blue(a0),convert_to_blue(a1),convert_to_blue(a2),convert_to_blue(a3),
			convert_to_blue(a4),convert_to_blue(a5),convert_to_blue(a6),convert_to_blue(a7)])'''
		bleu += sentence_bleu([convert_to_blue(a0),convert_to_blue(a1),convert_to_blue(a2),convert_to_blue(a3),
			convert_to_blue(a4),convert_to_blue(a5),convert_to_blue(a6),convert_to_blue(a7)], convert_to_blue(c), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sf.method3)
	print("Sentence level SARI")
	print(sari/l)
	print(keep/l)
	print(delete/l)
	print(add/l)
	print('Sentence level Bleu: ')
	print(bleu/l)
	print('Corpus Level Bleu: %f' % corpus_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sf.method3))

	print('Sentence level fkgl')
	print(f/l)
	print('Sentence level fre')
	print(fr/l)
	print('Average Length')
	print(avg_len/l)

elif dataset == 'Newsela':

	Complex = open('Newsela/Complex.txt', encoding='utf-8').read().split('\n')
	#DressLs = open('Newsela/Dress-Ls.txt', encoding='utf-8').read().split('\n')
	#Dress = open('Newsela/Dress.txt', encoding='utf-8').read().split('\n')
	#EncDecA = open('Newsela/EncDecA.txt', encoding='utf-8').read().split('\n')
	#Hybrid = open('Newsela/Hybrid.txt', encoding='utf-8').read().split('\n')
	#PBMTR = open('Newsela/PBMT-R.txt', encoding='utf-8').read().split('\n')
	#DMass = open('Newsela/DMass.txt', encoding='utf-8').read().split('\n')
	#S2SAllFA = open('Newsela/S2S-All-FA.txt', encoding='utf-8').read().split('\n')
	#EditNTS = open('Newsela/Editnts.txt', encoding='utf-8').read().split('\n')
	#EntPar = open('Newsela/EntPar.txt', encoding='utf-8').read().split('\n')
	Reference = open('Newsela/Reference.txt', encoding='utf-8').read().split('\n')
	#Baseline = open('base250.txt', encoding='utf-8').read().split('\n')
	Mine = open(n, encoding='utf-8').read().split('\n')
	#Mine = Baseline
	'''with open('metrics/Mine/'+n, "a") as file:
		l = len(Reference)
		for i in range(l):
			file.write(unicodeToAscii(EncDecA[i].lower()) + '\n')'''

	sari = 0.0
	keep = 0.0
	delete = 0.0
	add = 0.0
	bleu = 0.0
	f = 0.0
	fr = 0.0
	avg_len = 0
	candidate = []
	reference = []
	reference_sent = ''
	l = len(Reference)
	for i in range(l):
		avg_len += len(Mine[i].split(' ')) 
		a = unicodeToAscii(Reference[i].lower())
		#a = Reference[i]
		b = unicodeToAscii(Complex[i].lower())
		#b = Complex[i]
		c = unicodeToAscii(Mine[i].lower())
		#c = Dress[i]
		f += sentence_fkgl(c)
		fr += sentence_fre_final(c)
		candidate.append(convert_to_blue(c))
		reference.append([convert_to_blue(a)])
		s, k, d, ad = calculate(b, c, [a])
		sari += s
		keep += k
		delete += d
		add += ad
		bleu += sentence_bleu([convert_to_blue(a)], convert_to_blue(c), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sf.method3)
	print("Sentence level SARI metric")
	print(sari/l)
	print(keep/l)
	print(delete/l)
	print(add/l)
	print('Sentence level Bleu: ')
	print(bleu/l)
	print('Corpus Level Bleu: %f' % corpus_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sf.method3))
	print('Sentence level fkgl')
	print(f/l)
	print('Sentence level fre')
	print(fr/l)
	print('Average Length')
	print(avg_len/l)




