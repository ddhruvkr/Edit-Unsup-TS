from utils import *
print("Mode is")
print(config['operation'])
print(device)

if config['operation'] == 'train_lm':
	if config['lm_type'] == 'structural' or config['lm_type'] == 'structural_masked':
		from model.structural_decoder import DecoderGRU
		from test_structured import *
		from evaluate_structured import *
	elif config['lm_type'] == 'standard':
		from model.decoder import DecoderRNN
		from test import *
		from evaluate import *

	if config['lm_type'] == 'structural' or config['lm_type'] == 'standard':
		idf, unigram_prob, output_lang, tag_lang, dep_lang, train_simple_unique, valid_simple_unique, test_simple_unique, train_complex_unique, valid_complex_unique, test_complex_unique, output_embedding_weights, tag_embedding_weights, dep_embedding_weights = prepareData(config['embedding_dim'], 
		config['freq'], config['ver'], config['dataset'], config['operation'])
	#print(len(train_pairs))
	if config['lm_type'] == 'structural':
		decoder = DecoderGRU(config['hidden_size'], output_lang.n_words, tag_lang.n_words, dep_lang.n_words, config['num_layers'], 
			output_embedding_weights, tag_embedding_weights, dep_embedding_weights, config['embedding_dim'], config['tag_dim'], config['dep_dim'], config['dropout'], config['use_structural_as_standard']).to(device)
		train_pos, train_dep = load_syntax_file(train_simple_unique, 'train', config['lm_backward'])
		valid_pos, valid_dep = load_syntax_file(valid_simple_unique, 'valid', config['lm_backward'])
		test_pos, test_dep = load_syntax_file(test_simple_unique, 'test', config['lm_backward'])
		print('loaded pos, dep files')
	elif config['lm_type'] == 'standard':
		decoder = DecoderRNN(config['hidden_size'], output_lang.n_words, config['num_layers'], 
			output_embedding_weights, config['embedding_dim'], config['dropout']).to(device)

	for i in range(len(train_simple_unique)):
		train_simple_unique[i] = train_simple_unique[i].lower()
	for i in range(len(valid_simple_unique)):
		valid_simple_unique[i] = valid_simple_unique[i].lower()
	for i in range(len(test_simple_unique)):
		test_simple_unique[i] = test_simple_unique[i].lower()
	if config['lm_type'] == 'structural':
		trainIters(decoder, train_simple_unique, valid_simple_unique, test_simple_unique, output_lang, tag_lang, dep_lang, train_pos, train_dep, valid_pos, valid_dep, test_pos, test_dep)
	elif config['lm_type'] == 'standard':
		trainIters(decoder, train_simple_unique, valid_simple_unique, test_simple_unique, output_lang)


elif config['operation'] == "sample":
	if config['lm_type'] == 'structural':
		from model.structural_decoder import DecoderGRU
	elif config['lm_type'] == 'standard':
		from model.decoder import DecoderRNN

	idf, unigram_prob, output_lang, tag_lang, dep_lang, train_simple, valid_simple, test_simple, train_complex, valid_complex, test_complex, output_embedding_weights, tag_embedding_weights, dep_embedding_weights = prepareData(config['embedding_dim'], 
	config['freq'], config['ver'], config['dataset'], config['operation'])

	if config['lm_type'] == 'structural':
		lm_forward = DecoderGRU(config['hidden_size'], output_lang.n_words, tag_lang.n_words, dep_lang.n_words, config['num_layers'], 
			output_embedding_weights, tag_embedding_weights, dep_embedding_weights, config['embedding_dim'], config['tag_dim'], config['dep_dim'], config['dropout'], config['use_structural_as_standard']).to(device)
		lm_backward = DecoderGRU(config['hidden_size'], output_lang.n_words, tag_lang.n_words, dep_lang.n_words, config['num_layers'], 
			output_embedding_weights, tag_embedding_weights, dep_embedding_weights, config['embedding_dim'], config['tag_dim'], config['dep_dim'], config['dropout'], config['use_structural_as_standard']).to(device)

	elif config['lm_type'] == 'standard':
		lm_forward = DecoderRNN(config['hidden_size'], output_lang.n_words, config['num_layers'], 
			output_embedding_weights, config['embedding_dim'], config['dropout']).to(device)
		lm_backward = DecoderRNN(config['hidden_size'], output_lang.n_words, config['num_layers'], 
			output_embedding_weights, config['embedding_dim'], config['dropout']).to(device)

	from tree_edits_beam import *
	if config['set'] == 'valid':
		sample(valid_complex, valid_simple, output_lang, tag_lang, dep_lang, lm_forward, lm_backward, output_embedding_weights, idf, unigram_prob)
	elif config['set'] == 'test':
		sample(test_complex, test_simple, output_lang, tag_lang, dep_lang, lm_forward, lm_backward, output_embedding_weights, idf, unigram_prob)

else:
	print('incorrect operation')
