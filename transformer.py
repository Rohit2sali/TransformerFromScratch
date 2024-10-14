from Encoder import Encoder
from Decoder import Decoder
from tensorflow import maximum
from tensorflow.keras import Model 
from tensorflow.keras.layers import Dense
import tensorflow as tf

class TransformerModel(Model):
    def __init__(self, enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate, **kwargs):
        super().__init__(**kwargs)

        # set up the encoder
        self.encoder = Encoder(enc_vocab_size, enc_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate=rate)
  
        # set up the decoder
        self.decoder = Decoder(dec_vocab_size, dec_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate=rate)

        # define the final dense layer
        self.model_last_layer = Dense(dec_vocab_size)

    def padding_mask(self, input):
        # create mask which marks the zero padding values in the input by a 1
        seq = tf.cast(tf.math.equal(input, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
    
    def lookahead_mask(self, shape):
        # mask out future entries by marking them with 1.0
        mask = 1 - tf.linalg.band_part(tf.ones((shape, shape)), -1, 0)

        return mask
    
    # e class method call() to feed the relevant inputs into the encoder and decoder
    def call(self, encoder_input, decoder_input, training=False):

        # create padding mask to mask the encoder inputs and decoder output in the decoder
        enc_padding_mask = self.padding_mask(encoder_input)

        # create and combine padding and look-ahead mask to be fed into decoder 
        dec_in_padding_mask = self.padding_mask(decoder_input)
        dec_in_lookahead_mask = self.lookahead_mask(decoder_input.shape[1])
        dec_in_lookahead_mask = maximum(dec_in_padding_mask, dec_in_lookahead_mask)

        # fed the input into encoder 
        encoder_output = self.encoder(encoder_input, enc_padding_mask, training=training)

        # feed the encoder output in decoder
        decoder_output = self.decoder(decoder_input, encoder_output, dec_in_lookahead_mask, enc_padding_mask, training=training)

        # pass the decoder output through a final dense layer
        model_output = self.model_last_layer(decoder_output)

        return model_output
    