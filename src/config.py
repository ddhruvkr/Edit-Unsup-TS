model_config = {
    'clip': 50,
    'lr': 0.001,
    'num_steps': 87,
    'threshold': {'ls':0.8, 'dl':1.25, 'las':5.0, 'rl':1.25, 'pa': 1.25}, #Newsela -> {'ls':1.25, 'dl':1.25, 'las':1.25, 'rl':1.25, 'pa': 1.25}, Wikilarge ->{'ls':0.8, 'dl':1.25, 'las':5.0, 'rl':1.25, 'pa': 1.25}
    'epochs': 100,
    'set': 'test',
    'lm_name': 'Wikilarge/structured_lm_forward_300_150_0_4_freq5', #wikilarge -> Wikilarge/structured_lm_forward_300_150_0_4_freq5, newsela -> Newsela/structured_lm_forward_300_150_0_4
    'use_structural_as_standard': False,
    'lm_backward': False,
    'embedding_dim': 300,
    'tag_dim': 150,
    'dep_dim': 150,
    'hidden_size': 256,
    'num_layers': 2,
    'freq':0,
    'min_length': 100,
    'dataset': 'Wikilarge', #Wikilarge, Newsela
    'ver':'glove.6B.',
    'dropout':0.4,
    'batch_size':64,
    'print_every':100,
    'MAX_LENGTH': 85,
    'double_LM': False,
    'gpu': 0,
    'awd': False,
    'file_name': 'Wikilarge/output/simplifications_Wikilarge.txt',#Wikilarge/output/simplifications_Wikilarge.txt , Newsela/output/simplifications_Newsela.txt
    'fre': True,
    'SLOR': True,
    'beam_size': 1,
    'elmo': False,
    'min_length_of_edited_sent': 6,
    'lexical_simplification': True,
    'delete_leaves': True,
    'leaves_as_sent': True,
    'reorder_leaves': True,
    'check_min_length': True,
    'cos_similarity_threshold': 0.7, #WIKILARGE -> 0.7
    'cos_value_for_synonym_acceptance': 0.45, #Newsela ->0.5 WIKILARGE->0.45
    'min_idf_value_for_ls': 9,  #Wikilarge -> 9, NEwsela -> 11
    'sentence_probability_power': 0.5, #Wikilarge=0.5, Newsela->1.0
    'named_entity_score_power': 1.0,
    'len_power': 0.25, #Wikilarge=0.25, Newsela -> 1.0
    'fre_power': 1.0,
    'operation': 'sample' # or sample or train_lm,
}