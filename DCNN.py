"""
This project is to use a CNN-DCNN model to do semi-supervised or supervised binary text classification on Chinese
text.

It can be viewed as a Keras based implementation of the classification model in the paper ("Deconvolutional Paragraph
Representation Learning" by Yizhe Zhang, Dinghan Shen, Guoyin Wang, Zhe Gan, Ricardo Henao and Lawrence Carin,
NIPS 2017). Note that there are some differences on the layer settings and loss function which make the model easier to
train and fit the Chinese text.

I have implemented both a baseline purely CNN model and a semi-supervised CNN_DCNN model, you can separately train and
test on one model by selecting model type.

After some modification on input and layer settings, this project could be used to work on many other tasks like
text summarization and paragraph reconstruction.

How to run?
1. Train your selected model by providing pos.txt and neg.txt and save the model. See usage in train mode.
2. Do predictions on your test text based on your trained model. See usage in test mode.
3. You can also tune the parameters in the tuning mode. So far, it only supports hidden layer size tuning in CNN. You
can add more parameters to tune by modifying the function model_evaluate in DCNN.py easily

Usage: python3 DCNN.py \
-m train or test \
-k an integer for number of folds in cross validation for choosing hyperparameters \
-s hidden layer size \
-d /model_file/path/to/dict.p \

for train, need:
-i1 /raw_data/path/to/pos.txt \
-i2 /raw_data/path/to/neg.txt \
-o /model_file/path/to/output_model \
-e epochs

for test, need:
-a  /model_file/path/to/trained_model \
-t  /raw_data/path/to/test.txt \
-o /output_file/path/to/output_predictions \

for tuning, need:
-i1 /raw_data/path/to/pos.txt \
-i2 /raw_data/path/to/neg.txt \
-ps parameter sets (array of hidden_size you want to try)

Sample usage on Yuming's local environment, for training:
python DCNN.py \
-m train \
-i1 C:\\Users\\Umean\\Desktop\\Stratify\\pos.txt \
-i2 C:\\Users\\Umean\\Desktop\\Stratify\\neg.txt \
-mt DCNN \
-o C:\\Users\\Umean\\Desktop\\Stratify\\DCNN_model.h5

for testing(predicting):
python DCNN.py \
-m test \
-t C:\\Users\\Umean\\Desktop\\Stratify\\test.txt \
-a DCNN_model.h5 \
-mt DCNN \
-o C:\\Users\\Umean\\Desktop\\Stratify\\predictions.txt

for tuning:
python DCNN.py \
-m tuning \
-mt DCNN \
-i1 C:\\Users\\Umean\\Desktop\\Stratify\\pos.txt \
-i2 C:\\Users\\Umean\\Desktop\\Stratify\\neg.txt \
-ps [100, 300, 500]

Default convolution layer and embedding settings:
max_length = 20
max_words = 5000
embed_size = 300
filter_size = 300 (number of filters)
strides = 2
window_size = 4 (filter shape)
"""

import re
import sys
import argparse
from keras.models import Sequential
import keras.backend as K
from keras.layers import Conv2DTranspose, Lambda, Dropout
from keras.layers import Input, Embedding, Dense, Conv1D, Flatten
from keras.models import Model
from keras.models import load_model, clone_model
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.losses
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import scale
from sklearn.model_selection import StratifiedKFold
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import jieba
import pickle

EPOCHS = 10
BATCH_SIZE = 128
max_length = 20
max_words = 5000
embed_size = 300

filter_size = 300
strides = 2
window_size = 4


def read_text(filename):
    """
    read and clean the raw text
    """
    raw_text = []
    with open(filename, 'r', encoding='utf-8') as f:
        for sent in f.readlines():
            sent = re.sub("[`@#$^&*=|{}\':;\[\].<>/。_]", "", sent)
            sent = re.sub("na", "", sent)
            sent = re.sub("没有描述", "", sent)
            sent = sent.strip()
            if sent:
                raw_text.append(sent)
    return raw_text


def text_encoding(text):
    """
    Use jieba to transform sentences into Chinese words and encode them into a fixed length sequence
    """
    tokenizer = Tokenizer(num_words=max_words)
    for i in range(0, len(text)):
        text[i] = " ".join(jieba.cut(text[i], cut_all=False))
    tokenizer.fit_on_texts(text)
    text_encoded = tokenizer.texts_to_sequences(text)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    text_encoded = pad_sequences(text_encoded, maxlen=max_length)

    return text_encoded, word_index


def reconstuction_loss(y_true, y_pred):
    """
    Define the reconstruction lost
    """
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return K.mean(1 - K.sum((y_true * y_pred), axis=-1))


def CNN_model_define(filter_size, window_size, strides, hidden_size):
    """
    Define my CNN model with Keras
    """
    encoded_input = Input(shape=(max_length,), dtype='int32', name='encoded_input')
    embeded = Embedding(output_dim=embed_size, input_dim=max_words,
                        input_length=max_length, name='embedding_layer')(encoded_input)
    CNN1 = Conv1D(filters=filter_size, kernel_size=window_size, strides=strides, activation='relu')(embeded)
    hidden = Conv1D(filters=hidden_size, kernel_size=CNN1._keras_shape[1],
                    strides=strides, activation='relu')(CNN1)
    mid = Flatten()(hidden)
    mid = Dense(300)(mid)
    mid = Dropout(0.5)(mid)
    label_output = Dense(1, name='label_output', activation='sigmoid')(mid)
    model = Model(inputs=encoded_input, outputs=label_output)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def get_embedded_input(model, encoded_text):
    """
    Get embedding layer output from a CNN model as the input for CNN_DCNN model
    """
    embedding_layer_model = Model(inputs=model.input, outputs=model.get_layer('embedding_layer').output)
    return embedding_layer_model.predict(encoded_text)


def CNN_model_train(X, y, hidden_size, epochs):
    """
    Construct a CNN model with Keras and train
    :param hidden_size: an integer of hidden layer size.
    :param X: encoded text data for training
    :param y: corresponding label
    :param epochs: epochs for training
    :return model: a trained CNN Keras model
    """
    N = X.shape[0]
    y = y.reshape(N, 1)
    model = CNN_model_define(filter_size, window_size, strides, hidden_size)
    model.summary()
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    hist = model.fit(X, y, epochs=epochs, batch_size=BATCH_SIZE,
                     validation_split=0.2, callbacks=[early_stopping])
    with open("CNN_train_history.txt", "w") as f:
        print(hist.history, file=f)
    return model


def CNN_DCNN_model_define(filter_size, window_size, strides, hidden_size):
    """
    Define my CNN_DCNN model with Keras
    """
    def Conv1DTranspose(input_tensor, filters, kernel_size, activation, name=None, strides=2, padding='valid'):
        """
        Define a 1D deconvolution layer
        """
        x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
        x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1),
                            padding=padding, activation=activation)(x)
        x = Lambda(lambda x: K.squeeze(x, axis=2), name=name)(x)
        return x

    embedded_input = Input(shape=(max_length, embed_size), dtype='float32', name='embedded_input')
    CNN1 = Conv1D(filters=filter_size, kernel_size=window_size, strides=strides, activation='relu')(embedded_input)
    hidden = Conv1D(filters=hidden_size, kernel_size=CNN1._keras_shape[1],
                    strides=strides, activation='relu')(CNN1)
    mid = Flatten()(hidden)
    mid = Dense(300)(mid)
    label_output = Dense(1, name='label_output', activation='sigmoid')(mid)

    DCNN1 = Conv1DTranspose(input_tensor=hidden, filters=hidden_size, kernel_size=CNN1._keras_shape[1], strides=strides,
                            activation='relu')
    reconstruction_output = Conv1DTranspose(input_tensor=DCNN1, filters=filter_size, kernel_size=window_size, strides=strides,
                                     activation='relu', name='reconstruction_output')

    model = Model(inputs=embedded_input, outputs=[reconstruction_output, label_output])
    model.compile(optimizer='rmsprop',
                  loss={'reconstruction_output': reconstuction_loss, 'label_output': 'binary_crossentropy'},
                  loss_weights={'reconstruction_output': 0.01, 'label_output': 1.},
                  metrics={'label_output': 'accuracy'})
    return model

def CNN_DCNN_model_train(X, y, hidden_size, epochs):
    """
    Construct a CNN_DCNN_model with Keras and train
    :param hidden_size: an integer of hidden layer size.
    :param X: encoded text data for training
    :param y: corresponding label
    :param epochs: epochs for training
    :return model: a trained CNN_DCNN Keras model
    """
    model = CNN_DCNN_model_define(filter_size, window_size, strides, hidden_size)
    model.summary()
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    hist = model.fit(X, [X, y], epochs=epochs, batch_size=BATCH_SIZE,
                     validation_split=0.2, callbacks=[early_stopping])
    with open("CNN_DCNN_train_history.txt", "w") as f:
        print(hist.history, file=f)
    return model

def split_indices(y, num_fold):
    """
    Provide sets of train-test indices to split the raw data into several stratified folds
    :param y: labels of the raw_data, shape (N, )
    :param num_fold: an interger of the number of folds in Cross Validation.
    :return: a list of tuples (train_index, test_index) of length num_fold
                train_index: index of train data in each train-test set
                test_index: index of test data in each train-test set
    """
    skf = StratifiedKFold(n_splits=num_fold)
    N = y.shape[0]
    indices = skf.split(np.zeros(N), y)
    return indices


def model_evaluate(model_type, parameter_set, X, y, num_fold):
    """
    Evaluate the LSTM or rnn model by Stratified K-Folds Cross Validation.
    :param model_type: model type
    :param hidden_size: an integer of hidden layer size.
    :param X: encoded text data for training
    :param y: corresponding label
    :param num_fold: an interger of the number of folds in Cross Validation
    :return: a real value represents the accuracy.
    """
    best_cv_acc = 0
    best_parameter = 500
    for hidden_size in parameter_set:
        if model_type == "CNN":
            model_unfitted = CNN_model_define(filter_size, window_size, strides, hidden_size)
        if model_type == "DCNN":
            model_unfitted = CNN_DCNN_model_define(filter_size, window_size, strides, hidden_size)
        indices = split_indices(y, num_fold)
        cv_acc = []
        for train_index, test_index in indices:
            model = clone_model(model_unfitted)
            early_stopping = EarlyStopping(monitor='val_loss', patience=2)
            if model_type == "DCNN":
                model.compile(optimizer='rmsprop',
                              loss={'reconstruction_output': reconstuction_loss, 'label_output': 'binary_crossentropy'},
                              loss_weights={'reconstruction_output': 0.01, 'label_output': 1.},
                              metrics={'label_output': 'accuracy'})
                model.fit(X[train_index], [X[train_index], y[train_index]], epochs=EPOCHS,
                          batch_size=BATCH_SIZE, callbacks=[early_stopping])
            if model_type == "CNN":
                model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
                model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE,
                          validation_split=0.2, callbacks=[early_stopping])
            #label_model = Model(inputs=model.input, outputs=model.get_layer("label_output").output)
            cv_acc.append(model.evaluate(X[test_index], y[test_index]))
        mean_cv_acc = np.mean(np.array(cv_acc))
        if mean_cv_acc > best_cv_acc:
            best_cv_acc = mean_cv_acc
            best_parameter = hidden_size
    print("ACC: %.2f%%" % (best_cv_acc * 100))
    print("Best parameter: %d" % (best_parameter))
    return best_parameter


def model_predict(model, X, raw_text):
    """
    Do predictions based on the given trained LSTM or RNN model and the data features.
    :param model: the trained model
    :param X: embedded input data
    :param raw_text: raw text
    :return: a pandas DataFrame of raw_text, predict probabilities and predict labels.
    """
    N = X.shape[0]
    print("Predicting")
    label_model = Model(inputs=model.input, outputs=model.get_layer("label_output").output)
    y_prob = label_model.predict(X).reshape(N)
    y_pred = y_prob > 0.5
    predictions = pd.DataFrame({'raw text': raw_text, 'prob': y_prob, 'pred': y_pred})
    predictions = predictions[['raw text', 'prob', 'pred']]
    return predictions


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i1", "--pos_data", help="Where we read the positive text",
                           type=str, default=".", required=False)
    argparser.add_argument("-i2", "--neg_data", help="Where we read the negative text",
                           type=str, default=".", required=False)
    argparser.add_argument("-t", "--test_data", help="Where we read the test text",
                           type=str, default=".", required=False)
    argparser.add_argument("-a", "--trained_model", help="Where we read the trained model",
                           type=str, default=".", required=False)
    argparser.add_argument("-s", "--hidden_size", help="Size of hidden layer",
                           type=int, default=500, required=False)
    argparser.add_argument("-e", "--epoch_size", help="epochs",
                           type=int, default=8, required=False)
    argparser.add_argument("-m", "--mode", help="Train or test or tuning",
                           type=str, default=".", required=True)
    argparser.add_argument("-k", "--num_fold", help="Number of folds in cross validation",
                           type=int, default=5, required=False)
    argparser.add_argument("-o", "--output_file", help="Where we output predictions",
                           type=str, default=".", required=False)
    argparser.add_argument("-d", "--dictionary", help="Word index dictionary",
                           type=str, default="dict.p", required=False)
    argparser.add_argument("-mt", "--model_type", help="The method you want to use",
                           type=str, default=".", required=True)
    argparser.add_argument("-ps", "--parameter_set", help="parameter sets of tuning hidden size",
                           type=str, default="[100, 300, 500]", required=False)
    args = argparser.parse_args()

    global EPOCHS
    EPOCHS = args.epoch_size
    pos_data = args.pos_data
    neg_data = args.neg_data
    test_data = args.test_data
    mode = args.mode.lower()
    model_type = args.model_type.upper()
    hidden_size = args.hidden_size
    parameter_set = eval(args.parameter_set)
    num_fold = args.num_fold

    assert mode == "train" or mode == "test" or mode == "tuning", \
        "need to choose a mode: train, test(predict) or tuning"
    assert model_type in ["CNN", "DCNN"], "need to choose a model type: CNN or DCNN"

    if mode == "train":
        assert pos_data != "." and neg_data != ".", "need pos and neg data in test mode"
        pos_raw_text = read_text(pos_data)
        neg_raw_text = read_text(neg_data)
        all_text = pos_raw_text + neg_raw_text
        all_text_encoded, word_index = text_encoding(all_text)
        pickle.dump(word_index, open(args.dictionary, 'wb'))
        labels = np.array([1]*len(pos_raw_text) + [0]*len(neg_raw_text))
        idx = np.random.permutation(len(all_text_encoded))
        X, y = all_text_encoded[idx], labels[idx]
        if model_type == "CNN":
            # X = X[:10000]
            # y = y[:10000]
            model_CNN = CNN_model_train(X, y, hidden_size, EPOCHS)
            model_CNN.save(args.output_file)
        if model_type == "DCNN":
            # X = X[:10000]
            # y = y[:10000]
            print("Training a CNN model to get embeddings")
            model_CNN = CNN_model_train(X, y, hidden_size, EPOCHS)
            model_CNN.save('CNN_model.h5')
            embedded_input = get_embedded_input(model_CNN, X)
            print("Training CNN_DCNN")
            # print(embedded_input)
            model_CNN_DCNN = CNN_DCNN_model_train(embedded_input, y, hidden_size, EPOCHS)
            model_CNN_DCNN.save(args.output_file)

    if mode == "test":
        assert args.trained_model != ".", "a trained model is required in test mode"
        assert args.test_data != ".", "test data is required in test mode"
        if model_type == "CNN":
            model = load_model(args.trained_model)
            word_index = pickle.load(open(args.dictionary, 'rb'))
            raw_test_text = read_text(test_data)
            test_text = []
            for i in range(len(raw_test_text)):
                sent = [word_index[word] for word in jieba.cut(raw_test_text[i], cut_all=False) if word in word_index]
                sent = list(filter(lambda x: x < max_words, sent))
                test_text.append(sent)
            encoded_text = pad_sequences(test_text, maxlen=max_length)
            predictions = model_predict(model, encoded_text, raw_test_text)
            predictions.to_csv(args.output_file, index=False)

        if model_type == "DCNN":
            model_CNN = load_model('CNN_model.h5')
            model = load_model(args.trained_model, custom_objects={'reconstuction_loss': reconstuction_loss})
            word_index = pickle.load(open(args.dictionary, 'rb'))
            raw_test_text = read_text(test_data)
            test_text = []
            for i in range(len(raw_test_text)):
                sent = [word_index[word] for word in jieba.cut(raw_test_text[i], cut_all=False) if word in word_index]
                sent = list(filter(lambda x: x < max_words, sent))
                test_text.append(sent)
            encoded_text = pad_sequences(test_text, maxlen=max_length)
            embedded_input = get_embedded_input(model_CNN, encoded_text)
            predictions = model_predict(model, embedded_input, raw_test_text)
            predictions.to_csv(args.output_file, index=False)

    if mode == "tuning":
        if model_type == "DCNN":
            print("Unsupported yet")
            return 0
        pos_raw_text = read_text(pos_data)
        neg_raw_text = read_text(neg_data)
        all_text = pos_raw_text + neg_raw_text
        all_text_encoded, word_index = text_encoding(all_text)
        pickle.dump(word_index, open(args.dictionary, 'wb'))
        labels = np.array([1]*len(pos_raw_text) + [0]*len(neg_raw_text))
        idx = np.random.permutation(len(all_text_encoded))
        X, y = all_text_encoded[idx], labels[idx]
        # X = X[:100]
        # y = y[:100]
        model_evaluate(model_type, parameter_set, X, y, num_fold)
    return 0

if __name__ == '__main__':
    main()
