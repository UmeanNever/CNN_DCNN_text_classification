# CNN_DCNN_text_classification
*by Yuming Yang*

This project is to use a CNN-DCNN model to do semi-supervised or supervised binary text classification on Chinese
text.

It can be viewed as a Keras based implementation of the classification model in the paper **("Deconvolutional Paragraph
Representation Learning" by Yizhe Zhang, Dinghan Shen, Guoyin Wang, Zhe Gan, Ricardo Henao and Lawrence Carin,
NIPS 2017)**. Note that there are some differences on the layer settings and loss function which make the model easier to
train and fit the Chinese text.

I have implemented both a baseline purely CNN model and a semi-supervised CNN_DCNN model, you can separately train and
test on one model by selecting model type.

After some modification on input and layer settings, this project could be used to work on many other tasks like
text summarization and paragraph reconstruction.

## How to run?

1. Train your selected model by providing pos.txt and neg.txt and save the model. See usage in train mode.
2. Do predictions on your test text based on your trained model. See usage in test mode.
3. You can also tune the parameters in the tuning mode. So far, it only supports hidden layer size tuning in CNN. You
can add more parameters to tune by modifying the function model_evaluate in DCNN.py easily

**(If you just want to see an example, try directly do predictions using my trained model and test data. See Sample Usage for testing for datails)**

## Usage: 
```
python3 DCNN.py \
-m train or test \
-k an integer for number of folds in cross validation for choosing hyperparameters \
-s hidden layer size \
-d /model_file/path/to/dict.p \
```

for train, need:
```
-i1 /raw_data/path/to/pos.txt \
-i2 /raw_data/path/to/neg.txt \
-o /model_file/path/to/output_model \
-e epochs
```

for test, need:
```
-a  /model_file/path/to/trained_model \
-t  /raw_data/path/to/test.txt \
-o /output_file/path/to/output_predictions \
```

for tuning, need:
```
-i1 /raw_data/path/to/pos.txt \
-i2 /raw_data/path/to/neg.txt \
-ps parameter sets (array of hidden_size you want to try)
```

## Sample usage on Yuming's local environment
for training:
```
python DCNN.py \
-m train \
-i1 C:\\Users\\Umean\\Desktop\\Stratify\\pos.txt \
-i2 C:\\Users\\Umean\\Desktop\\Stratify\\neg.txt \
-mt DCNN \
-o C:\\Users\\Umean\\Desktop\\Stratify\\DCNN_model.h5
```
for testing(predicting):
```
python DCNN.py \
-m test \
-t C:\\Users\\Umean\\Desktop\\Stratify\\test.txt \
-a DCNN_model.h5 \
-mt DCNN \
-o C:\\Users\\Umean\\Desktop\\Stratify\\predictions.txt
```
for tuning:
```
python DCNN.py \
-m tuning \
-mt DCNN \
-i1 C:\\Users\\Umean\\Desktop\\Stratify\\pos.txt \
-i2 C:\\Users\\Umean\\Desktop\\Stratify\\neg.txt \
-ps [100, 300, 500]
```

## Default convolution layer and embedding settings:
```
max_length = 20
max_words = 5000
embed_size = 300
filter_size = 300 (number of filters)
strides = 2
window_size = 4 (filter shape)
```
