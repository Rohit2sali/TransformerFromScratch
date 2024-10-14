from pickle import load, dump, HIGHEST_PROTOCOL
from numpy.random import shuffle
from numpy import savetxt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import convert_to_tensor, int64
import tensorflow as tf
import numpy as np
tf.config.run_functions_eagerly(True) # if you want to print any tensor object this will let it do 

""" In this file we are tokenizing the data for Encoder and Decoder. We will return trainX which is tokenized tensor for Encoder and trainY which is tokenized 
    tensor for the Decoder, We also return sequence length and vocabulary size of Encoder and Decoder."""

  
class PerpareDateset:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_sentences = 500 # No of sentences to include in dataset
        self.train_split = 0.9 # Ratio of training data split
        self.val_split = 0.1 # Ratio of the validation data split

    # fit a Tokenizer
    def create_tokenizer(self, dataset):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(dataset)
        return tokenizer
    
    def find_seq_length(self, dataset):
        return max(len(seq.split()) for seq in dataset)
    
    def find_vocab_size(self, tokenizer, dataset):
        tokenizer.fit_on_texts(dataset)
        return len(tokenizer.word_index) + 1
    
    def encode_pad(self, dataset, tokenizer, seq_length):
        """If the maximum size your encoder or Decoder can take is larger than the sequence length of the input sentence 
         then you have to pad the tokens. Padding is done by adding some dummy tokens to your input data.
         such as you input sentence is of length 5 and the maximum your models takes is 7 so you will pad the sentence in this way
         [a, b, c, d, e] becomes [a, b, c, d, e, <PAD>, <PAD>] """
        x = tokenizer.texts_to_sequences(dataset)
        x = pad_sequences(x, maxlen=seq_length, padding='post')
        x = convert_to_tensor(x, dtype=int64)
        return x
    
    def save_tokenizer(self, tokenizer, name):
        """The tokenizer converts raw text into token IDs that the model understands. If you don't save the tokenizer, you would
           need to recreate it each time you want to preprocess text for inference or further training.
           Saving ensures that the same vocabulary and tokenization logic are used consistently across different runs,
           preventing discrepancies in how input text is handled."""
        
        with open(name + '_tokenizer.pkl', 'wb') as handle:
            dump(tokenizer, handle, protocol=HIGHEST_PROTOCOL)
    
    def call(self, filename, **kwargs):
        """this function is used for text Normalization such as lowercasing, stemming, or lemmatization, it is applied to ensure consistency """
        clean_dataset = load(open(filename, 'rb'))

        # reduce dataset size
        dataset = clean_dataset[:self.n_sentences, : ]

        # include start and end of string tokens
        for i in range(dataset[:, 0].size):
            dataset[i, 0] = "<START>" + dataset[i, 0] + "<EOS>"
            dataset[i, 1] = "<START>" + dataset[i, 1] + "<EOS>"

        shuffle(dataset)

        train = dataset[:int(self.n_sentences * self.train_split)]
        val = dataset[int(self.n_sentences * self.train_split): int(self.n_sentences * (1-self.val_split))]
        test = dataset[int(self.n_sentences * (1 - self.val_split)):]

        # prepare Tokenizer for the encoder input
        enc_tokenizer = self.create_tokenizer(train[:, 0])
        enc_seq_length = self.find_seq_length(train[:, 0])
        enc_vocab_size = self.find_vocab_size(enc_tokenizer, train[:, 0])

        # prepare tokenizer for decoder input 
        dec_tokenizer = self.create_tokenizer(train[:, 1])
        dec_seq_length = self.find_seq_length(train[:, 1])
        dec_vocab_size = self.find_vocab_size(dec_tokenizer, train[:, 1])

        # Encode and pad the training input
        trainX = self.encode_pad(train[:, 0], enc_tokenizer, enc_seq_length) # the shape of trainX will be (n_sentences, enc_seq_length)
        trainY = self.encode_pad(train[:, 1], dec_tokenizer, dec_seq_length) # the shape of trainY will be (n_sentences, dec_seq_length)
    
        # Encode and pad the validation input
        valX = self.encode_pad(val[:, 0], enc_tokenizer, enc_seq_length)
        valY = self.encode_pad(val[:, 1], dec_tokenizer, dec_seq_length)

        # Save the encoder tokenizer
        self.save_tokenizer(enc_tokenizer, 'enc')

        # Save the decoder tokenizer
        self.save_tokenizer(dec_tokenizer, 'dec')

        # Save the testing dataset into a text file
        savetxt('test_dataset.txt', test, fmt='%s')

        return (trainX, trainY, valX, valY, train, val, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size)

  