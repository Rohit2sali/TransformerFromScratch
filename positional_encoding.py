import tensorflow as tf
from tensorflow import convert_to_tensor, string 
from tensorflow.keras.layers import TextVectorization, Embedding, Layer
from tensorflow.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
tf.config.run_functions_eagerly(True) # if you want to print any tensor object this will let it do

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or '3' to suppress more warnings


class PositionEmbeddingFixedWeights(Layer):
    def __init__(self, seq_length, vocab_size, output_dim, **kwargs):
        super().__init__(**kwargs)
        word_embedding_matrix = self.get_position_encoding(vocab_size, output_dim)
        pos_embedding_matrix = self.get_position_encoding(seq_length, output_dim)
        
        self.word_embedding_layer = Embedding(input_dim=vocab_size, output_dim=output_dim,
                                              weights=[word_embedding_matrix], trainable = False)
        
        self.position_embedding_layer = Embedding(input_dim=seq_length, output_dim=output_dim,
                                                  weights=[pos_embedding_matrix], trainable=False)
        

    def get_position_encoding(self, seq_len, d, n=10000):
        p = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                p[k, 2*i] = np.sin(k / denominator)
                p[k, 2*i + 1] = np.cos(k / denominator)
        return p 
    

    def call(self, inputs):
        position_indices = tf.range(tf.shape(inputs)[-1])
        
        embedded_words = self.word_embedding_layer(inputs)

        embedded_indices = self.position_embedding_layer(position_indices)
       
        return embedded_words + embedded_indices
        # The output will of the size ( seq_length, output_len )
    