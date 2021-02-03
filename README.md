# embedding_augmentation

Implementation for ICASSP 2021 paper 

**"Towards Efficiently Diversifying Dialogue Generation via Embedding Augmentation"**

###Requirements

1. python >= 3.6.0
2. torch >= 1.1.0
3. fasttext == 0.9.2

###How to run

####1. Prepare the data

PersonaChat: You can obtain it from ParlAI, taking [this](https://github.com/facebookresearch/ParlAI/tree/personachat/projects/personachat)
as a reference. In order to obtain the 131k training sample, please the the `train_both_original.txt` as the training set.

DailyDialog: Download it from [this site](http://yanran.li/dailydialog).

Put both datasets under `./datasets` directory.

You also need to download the 6B [GLoVe embedding vectors](http://nlp.stanford.edu/data/glove.6B.zip), unzip it and put 
the 300d file under `./glove`

To calculate METEOR metrics, you need to download the [software package](https://www.cs.cmu.edu/~alavie/METEOR/) and 
extract it under `./metrics/`. If you do not need it, you can comment L626 in `./metrics.py`

####2. Train a FastText model

`python train_fasttext.py [YOUR DATA TXT FILE] [YOUR OUTPUT MODEL NAME]`

e.g.

`python train_fasttext.py train_self_original.txt persona_cbow_model.bin`

then put the trained model under `./fasttext_model`

####3. Run the training

Please take the descriptions in `config.py` as a reference.

Before formally running the experiment, you can try `run_dummy.sh` for testing.

We also provide a example shell script `train_persona.sh` to run the training using Transformer with embedding 
augmentation on PersonaChat dataset or `train_dialog.sh` to run the training on DailyDialog dataset.

