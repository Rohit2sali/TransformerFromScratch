from tensorflow import math, matmul, reshape, shape, transpose, cast, float32
from tensorflow.keras.layers import Dense, Layer
from keras.activations import softmax

class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, queries, keys, values, d_k, mask=None):
        # scoring the queries against the keys after transposing the latter, and scaling 
        scores = matmul(queries, keys, transpose_b=True) / math.sqrt(cast(d_k, float32))

        # apply mast to attention scores  
        if mask is not None:
            scores += -1e9 * mask

        # computing weights by softmax opration   
        weights = softmax(scores)

        # computing the attention by a weigthed sum of value vectors 
        return matmul(weights, values)
    
    
class MultiHeadAttention(Layer):

    def __init__(self, h, d_k, d_v, d_model, **kwargs): # h is equal to no of heads 
        super().__init__(**kwargs)
        self.attention = DotProductAttention() # scaled dot product attention 
        self.heads = h # number of attention heads to use
        self.d_k = d_k # dimensionality is linearly prejected queries and keys 
        self.d_v = d_v # dimensionality of linearly projected values 
        self.d_model = d_model # Dimensionality of the model
        self.W_q = Dense(d_k) # learned projection matrix of queries , it will be of shape (d_model, d_model / heads)
        self.W_k = Dense(d_k) # learned projention matrix of keys, it will be of shape (d_model, d_model / heads)
        self.W_v = Dense(d_v) # learned projection matrix of values, it will be of shape (d_model, d_model / heads)
        self.W_o = Dense(d_model) # learned projection matrix of multihead output, it will be of shape (d_model, d_model)

    def reshape_tensor(self, x, heads, flag):
        if flag:
            # tensor shape after reshaping and transposing : (batch_size, heads, seq_length, -1)
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], heads, -1))
            x = transpose(x, perm=(0, 2, 1, 3))
        else:
            # reverting the reshaping and transposing operations: (batch_size, seq_len, d_k)
            x = transpose(x, perm=(0, 2, 1, 3))
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], self.d_k))
        return x
    
    def call(self, queries, keys, values, mask=None):
        # rearrange the queries to be able to compute all heads in parrallel
        q_reshaped = self.reshape_tensor(self.W_q(queries), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange the keys to be able to compute all heads in parallel
        k_reshaped = self.reshape_tensor(self.W_k(keys), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange the values to be able to compute all heads in parallel
        v_reshaped = self.reshape_tensor(self.W_v(values), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Compute the multi-head attention output using the reshaped queries, keys, and values

        output= self.attention(q_reshaped, k_reshaped, v_reshaped, d_k=self.d_k, mask=mask)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange back the output into concatenated form
        output_reshaped = self.reshape_tensor(output, self.heads, False)
        # Resulting tensor shape: (batch_size, input_seq_length, d_v)

        # Apply one final linear projection to the output to generate the multi-head attention. Resulting tensor shape: (batch_size, input_seq_length, d_model)
        return self.W_o(output_reshaped)
        # This is will give output of shape (dimension of queries, input_seq_length, dimension_model)
    