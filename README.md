# embedding_augmentation

Implementation for ICASSP 2021 paper 

**Towards Efficiently Diversifying Dialogue Generation via Embedding Augmentation**

### Requirements

1. python >= 3.6.0
2. torch >= 1.1.0
3. fasttext == 0.9.2

### How to run

#### 1. Prepare the data and environment

PersonaChat: You can obtain it from ParlAI, taking [this](https://github.com/facebookresearch/ParlAI/tree/personachat/projects/personachat)
as a reference. In order to obtain the 131k training sample, please the the `train_both_original.txt` as the training set.
You can put the dataset under `./datasets/personachat`

DailyDialog: Download it from [this site](http://yanran.li/dailydialog). Unzip it and put the `.txt` files under 
`./datasets/dailydialog`, including the train/valid/test files `dialogues_train.txt`, `dialogues_validation.txt`, 
`dialogues_test.txt`. Then use the script `dailydialog.py` under this directory to convert these three files into three 
new files, `train_data.txt`, `valid_data.txt`, and `test_data.txt`, whose format can be recognized by our scripts. 

You also need to download the 6B [GLoVe embedding vectors](http://nlp.stanford.edu/data/glove.6B.zip), unzip it and put 
the 300d file under `./glove`

Perl is needed, 

To calculate METEOR metrics, you need to download the [software package](https://www.cs.cmu.edu/~alavie/METEOR/) and 
extract it under `./metrics/`. If you do not need it, you can comment L626 in `./metrics.py`

#### 2. Train a FastText model

`python train_fasttext.py [YOUR DATA TXT FILE] [YOUR OUTPUT MODEL NAME]`

e.g.

`python train_fasttext.py train_self_original.txt persona_cbow_model.bin`

then put the trained model under `./fasttext_model`

#### 3. Run the training

Please take the descriptions in `config.py` as a reference.

Before formally running the experiment, you can try `run_dummy.sh` for testing.

We also provide a example shell script `train_persona.sh` to run the training using Transformer with embedding 
augmentation on PersonaChat dataset or `train_dailydialog.sh` to run the training on DailyDialog dataset.

