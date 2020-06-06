model_config = {
    'clip': 50,
    'lr': 0.001,
    'num_steps': 87,
    'threshold': {'ls':1.25, 'dl':1.25, 'las':1.25, 'rl':1.25, 'pa': 1.25},
    'epochs': 100,
    'set': 'test',
    'lm_name': 'Newsela/structured_lm_forward_300_150_0_4', #wikilarge -> Wikilarge/structured_lm_forward_300_150_0_4_freq5
    'use_structural_as_standard': False,
    'lm_backward': False,
    'embedding_dim': 300,
    'tag_dim': 150,
    'dep_dim': 150,
    'hidden_size': 256,
    'num_layers': 2,
    'freq':0,
    'min_length': 100,
    'dataset': 'Newsela', #Wikilarge
    'ver':'glove.6B.',
    'dropout':0.4,
    'batch_size':64,
    'print_every':100,
    'MAX_LENGTH': 85,
    'double_LM': False,
    'gpu': 0,
    'awd': False,
    'file_name': 'Newsela/output/simplifications_Newsela.txt',#Wikilarge/output/testing_ls_ro_para_lm_2.txt
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
    'cos_value_for_synonym_acceptance': 0.5, #Newsela ->0.5 WIKILARGE->0.45
    'min_idf_value_for_ls': 11,  #Wikilarge -> 9, NEwsela -> 11
    'sentence_probability_power': 1.0, #Wikilarge=0.5
    'named_entity_score_power': 1.0,
    'len_power': 1.0, #Wikilarge=0.25
    'fre_power': 1.0,
    'operation': 'sample' # or sample or train_lm,
}