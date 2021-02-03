from sys import argv

import fasttext

input_file = argv[1]
output_model = argv[2]

model = fasttext.train_unsupervised(input_file, model='cbow', epoch=50)
model.save_model(output_model)
print('Saved trained FastText model in ' + output_model)