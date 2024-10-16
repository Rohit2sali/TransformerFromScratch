from tensorflow.keras.layers import LayerNormalization, Layer, Dense, ReLU, Dropout
from multihead_attention import MultiHeadAttention
from positional_encoding import PositionEmbeddingFixedWeights
import tensorflow as tf
 

# implement the Add and Norm layer 
class AddNormalization(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = LayerNormalization() # layer normalization layer
  
    def call(self, x, sublayer_x):
        # the sub layer input and output need to be of the same shape to be summed
        add = x + sublayer_x

        # apply layer normalization to sum
        return self.layer_norm(add)
    
# Implementing the Feed-Forward Layer
class FeedForward(Layer):
    def __init__(self, d_ff, d_model, **kwargs):
        super().__init__(**kwargs)
        self.fully_connected1 = Dense(d_ff) # First fully connected layer
        self.fully_connected2 = Dense(d_model) # Second fully connected layer
        self.activation = ReLU() # ReLU activation layer
    def call(self, x):
        # The input is passed into the two fully-connected layers, with a ReLU in between
        x_fc1 = self.fully_connected1(x)
        return self.fully_connected2(self.activation(x_fc1))
    
class EncoderLayer(Layer):
    def __init__(self, sequence_length, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
        super().__init__(**kwargs)
        
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()
    
    def build(self, input_shape):
        # You can define custom weights or logic based on the input shape here.
        # In this case, no extra weight tensors are needed beyond the ones in sub-layers,
        # so you don't need to define any custom weights manually.
        super().build(input_shape)  # This ensures the base class build() is called

    def call(self, x, padding_mask, training):
        # Multi-head attention layer
        multihead_output = self.multihead_attention(x, x, x, padding_mask)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in a dropout layer
        multihead_output = self.dropout1(multihead_output, training=training)
        # Followed by an Add & Norm layer

        addnorm_output = self.add_norm1(x, multihead_output)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Followed by a fully connected layer
        feedforward_output = self.feed_forward(addnorm_output)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in another dropout layer
        feedforward_output = self.dropout2(feedforward_output, training=training)
        # Followed by another Add & Norm 
        
        return self.add_norm2(addnorm_output, feedforward_output)

# implementing the encoder 
class Encoder(Layer):
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate, **kwargs):
        super().__init__(**kwargs)
        self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size, d_model)
        self.dropout = Dropout(rate)
        self.encoder_layer = [EncoderLayer(sequence_length, h, d_k, d_v, d_model, d_ff, rate=rate) for _ in range(n)]
    

    @tf.function
    def call(self, input_sentence, padding_mask, training):
        # generate the positional encoding 
        pos_encoding_output = self.pos_encoding(input_sentence)
        # expected output shape = (batch_size, seq_length, d_model)

        # Add a dropout layer
        x = self.dropout(pos_encoding_output, training=training)

        # pass on the positional encoded values to each encoded layer
        for i, layer in enumerate(self.encoder_layer):
            x = layer(x, padding_mask=padding_mask, training=training)

        return x
    