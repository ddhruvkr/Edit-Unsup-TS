# Edit-Unsup-TS

Code for our paper "[Iterative Edit-Based Unsupervised Sentence Simplification](https://www.aclweb.org/anthology/2020.acl-main.707.pdf)" accepted at ACL 2020.

Please cite this paper if you use our code or system output.

```
@inproceedings{kumar-etal-2020-iterative,
    title = "Iterative Edit-Based Unsupervised Sentence Simplification",
    author = "Kumar, Dhruv  and
      Mou, Lili  and
      Golab, Lukasz  and
      Vechtomova, Olga",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    publisher = "Association for Computational Linguistics",
    pages = "7918--7928",
}
```

The code is written in Python 3.7.6 and Pytorch 1.4.0.

# Training and running the model

You need to run the main.py file.

The different configurations in the code can be controlled by the config.py file.

To initially train the language model, set the "operation" parameter in the config file to "train_lm". This will train a syntax-aware language model. If you wish to use a standard language model, set the parameter "use_structural_as_standard" as True.

To run the simplification algorithm, set the parameter "operation" to "sample".

The syntax-aware language models for the Newsela and Wikilarge datasets, the trained word2vec model for Wikilarge (used in lexical simplification) and the simplified outputs from our models for the Wikilarge dataset can be found [[here]](https://drive.google.com/drive/folders/1We3YeS6O9iReXvcxG4XKx0pMO6bIggUR?usp=sharing)
For the simplified outputs for the Newsela dataset, please reach out to me at ddhruvkr@gmail.com. This is because the Newsela dataset is not publically available and only available via a contract with Newsela.

The language models should be put in src/Newsela and src/Wikilarge respectively. The trained word2vec model on the Wikilarge dataset, should be put in src/Wikilarge/Word2vec folder.

The POS and DEP tags for the sentences in Newsela and Wikilarge are precomputed and can be found in the folders src/Newsela and src/Wikilarge folders as well. If these files are not present, the code will generate these files.

You will also need to have a CoreNLP Server running on port 9000. You can download the package from [[here]](https://stanfordnlp.github.io/CoreNLP/download.html)


## Metric calculation

Make sure to calculate all metrics as CORPUS level and not SENTENCE level.

Use the calculate_scores.py file to calculate the CORPUS level BLEU scores for both the datasets.

The script for calculating CORPUS level SARI scores can be found in the JOSHUA package [[here]](https://github.com/XingxingZhang/dress/tree/master/experiments/evaluation/SARI)

The scripts for CORPUS level FKGL and FRE can be found in the folder readability_py2. It is important to run this in Python 2.7 environment to get the exact scores of those reported in all the previous work.


