# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


class CrossAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys, keys_mask):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, key_vec_size)
          keys_mask: Tensor shape (batch_size, num_keys)
            1s where there's real input, 0s where there's padding

        Outputs:
          att_vec_for_keys: Tensor shape (batch_size, num_keys, hidden_size).
          att_vec_for_values: Tensor shape (batch_size, num_values, hidden_size).
        """
        with vs.variable_scope("CrossAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_matrix = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)

            values_mask_matrix = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            keys_mask_matrix = tf.expand_dims(keys_mask, 2)    # shape (batch_size, num_keys, 1)

            _, attn_dist_values = masked_softmax(attn_matrix, values_mask_matrix, 2) # shape (batch_size, num_keys, num_values). take softmax over values
            _, attn_dist_keys = masked_softmax(attn_matrix, keys_mask_matrix, 1) # shape (batch_size, num_keys, num_values). take softmax over keys
            
            attn_dist_keys = tf.transpose(attn_dist_keys, perm=[0, 2, 1]) # shape (batch_size, num_values, num_keys)

            att_vec_for_keys = tf.matmul(attn_dist_values, values) # shape (batch_size, num_keys, value_vec_size)
            att_vec_for_values = tf.matmul(attn_dist_keys, keys)   # shape (batch_size, num_values, value_vec_size)

            # Apply dropout
            att_vec_for_keys = tf.nn.dropout(att_vec_for_keys, self.keep_prob)
            att_vec_for_values = tf.nn.dropout(att_vec_for_values, self.keep_prob)

            return att_vec_for_keys, att_vec_for_values

# Not Implemented Completely
# class SelfAttnRnet(object):
#     """Module for RNEt attention (Extension of Basic Attention).

#     Note: in this module we use the terminology of "keys" and "values" (see lectures).
#     In the terminology of "X attends to Y", "keys attend to values".

#     In the baseline model, the keys are the context hidden states
#     and the values are the question hidden states.

#     We choose to use general terminology of keys and values in this module
#     (rather than context and question) to avoid confusion if you reuse this
#     module with other inputs.
#     """

#     def __init__(self, keep_prob, key_vec_size, value_vec_size):
#         """
#         Inputs:
#           keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
#           key_vec_size: size of the key vectors. int
#           value_vec_size: size of the value vectors. int
#         """
#         self.keep_prob = keep_prob
#         self.key_vec_size = key_vec_size
#         self.value_vec_size = value_vec_size

#     def build_graph(self, values, values_mask, keys):
#         """
#         Keys attend to values.
#         For each key, return an attention distribution and an attention output vector.

#         Inputs:
#           values: Tensor shape (batch_size, num_values, value_vec_size).
#           values_mask: Tensor shape (batch_size, num_values).
#             1s where there's real input, 0s where there's padding
#           keys: Tensor shape (batch_size, num_keys, value_vec_size)

#         Outputs:
#           attn_dist: Tensor shape (batch_size, num_keys, num_values).
#             For each key, the distribution should sum to 1,
#             and should be 0 in the value locations that correspond to padding.
#           output: Tensor shape (batch_size, num_keys, hidden_size).
#             This is the attention output; the weighted sum of the values
#             (using the attention distribution as weights).
#         """
#         with vs.variable_scope("SelfAttnRnet"):

#             # Calculate attention distribution
#             values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
#             attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
#             attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
#             _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

#             # Use attention distribution to take weighted sum of values
#             output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

#             # Apply dropout
#             output = tf.nn.dropout(output, self.keep_prob) # shape (batch_size, num_keys, value_vec_size)

#             W1  = tf.get_variable(name = "W1", shape = [self.key_vec_size, 1], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
#             W1  = tf.get_variable(name = "W2", shape = [self.key_vec_size, 1], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())

#             return attn_dist, output

class BidirectionalAttn(object):
    """Module for Bidirectional attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys, keys_mask):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, key_vec_size)
          keys_mask: Tensor shape (batch_size, num_keys)
            1s where there's real input, 0s where there's padding

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BidirectionalAttn"):
            # Divide the weight matrix in 3 parts
            weights_sim1 =  tf.get_variable(name = "weights_sim1", shape = [self.key_vec_size, 1], dtype = tf.float32, initializer = tf.random_normal_initializer())
            weights_sim2 =  tf.get_variable(name = "weights_sim2", shape = [self.value_vec_size, 1], dtype = tf.float32, initializer = tf.random_normal_initializer())
            weights_sim3 =  tf.get_variable(name = "weights_sim3", shape = [self.key_vec_size], dtype = tf.float32, initializer = tf.random_normal_initializer())

            # Obtain Similarity Matrix sim_matrix/S where S_ij = w.T*[c;q;c o q]
            # c: context/keys, q: question/values, w = [w1,w2,w3]
            # values shape: (batch_size, num_values, value_vec_size)
            # keys shape:   (batch_size, num_keys, value_vec_size)
            batch_size = tf.shape(values)[0] # batch_size
            num_values = tf.shape(values)[1] # question_len
            num_keys   = tf.shape(keys)[1]   # context_len
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)

            # Part 1
            S1 = tf.reshape(tf.matmul(tf.reshape(keys, [-1, self.key_vec_size]), weights_sim1),[-1, num_keys, 1]) # shape : (batch_size, num_keys, 1)
            
            # Part 2
            S2 = tf.reshape(tf.matmul(tf.reshape(values, [-1, self.value_vec_size]), weights_sim2),[-1, num_values, 1]) # shape : (batch_size, num_values, 1)
            S2 = tf.transpose(S2, perm=[0, 2, 1])   # shape : (batch_size, 1,   num_values). Transposed for Broadcasting

            # Part 3: GPU efficient version
            weights_sim3 = tf.expand_dims(tf.expand_dims(weights_sim3,0),1)  # make it (1, 1, key_vec_size)
            ctile = tf.multiply(keys, weights_sim3) #(batch_size, num_keys, value_vec_size)
            S3 = tf.matmul(ctile, values_t)

            # Final sim_matrix/S obtained via broadcasting and adding 3 terms: S_shape->(batch_size, num_keys, num_values)
            sim_matrix = S1+S2+S3

            # Calculate mask same shape as S based on keys_mask & values_mask
            sim_mask = tf.expand_dims(keys_mask, 2)*tf.expand_dims(values_mask, 1)  # (batch_size, num_keys, num_values)

            # Context-to-Question (C2Q) Attention
            # Calculate attention distribution
            _, attn_dist = masked_softmax(sim_matrix, sim_mask, 2)

            # Use attention distribution to take weighted sum of values: a is the attended query vector
            att_vec = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Question-to-Context (Q2C) Attention
            m = tf.reduce_max(sim_matrix, 2) # shape(batch_size, num_keys)
            _, beta_dist = masked_softmax(m, keys_mask, 1) # shape (batch_size, num_keys)
            beta_dist = tf.expand_dims(beta_dist, 1)       # shape (batch_size, 1, num_keys)
            # Use attention distribution to take weighted sum of values: c_prime is Q2C attention vector
            c_prime = tf.matmul(beta_dist, keys) # shape (batch_size, 1, key_vec_size)

            # Apply dropout
            att_vec = tf.nn.dropout(att_vec, self.keep_prob)
            c_prime = tf.nn.dropout(c_prime, self.keep_prob)

            return att_vec, c_prime

class CoAttn(object):
    """Module for co-attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size


    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        
        

        with vs.variable_scope("CoAttn"):
            
            #########################################################################
            # Introduce a non-linear projection layer on top of the question encoding
            # to allow for varation between question and document encoding space
            W_q = tf.get_variable(name="W_q", \
                                shape=[2*self.key_vec_size,2*self.key_vec_size],\
                                initializer=tf.contrib.layers.xavier_initializer(),\
                                dtype=tf.float32)

            W_q = tf.expand_dims(tf.ones([tf.shape(values)[0],1]), 1) * W_q

            b_q = tf.Variable(tf.constant(0.0,\
                                shape=[2*self.key_vec_size,]),\
                                dtype=tf.float32,\
                                name='b_q')
            
            Q = tf.nn.tanh(tf.matmul(values,W_q)+b_q)
            #########################################################################

            #########################################################################
            # Coattention Encoder 
            
            # L --> affinity matrix
            l = tf.matmul(keys,tf.transpose(Q,perm=[0,2,1]))
            
            # mask affinity matrix, to not care about padded values
            l_mask = tf.expand_dims(values_mask, 1)
            
            # a_q --> attention weights across document for each word in the question
            _, a_q = masked_softmax(l, l_mask, 2)
            
            # a_d --> attention weights across question for each word in the document
            a_d = tf.nn.softmax(tf.transpose(l,perm=[0, 2, 1]))
            
            # compute attention context of the document in light of each word of the question
            c_q = tf.matmul(tf.transpose(a_q,perm=[0,2,1]),keys)
            
            # parallel computation of q*a_d & c_q*a_q
            c_d = tf.matmul(tf.transpose(a_d,perm=[0,2,1]),tf.concat([Q,c_q],2))
            
            # d_c_d as concatinated d;c_d as inpit for the Bi-LSTM
            d_c_d = tf.concat([keys,c_d],axis=2)
            
            d_c_d_length = tf.cast(\
                            tf.reduce_sum(\
                                tf.sign(\
                                    tf.reduce_max(\
                                        tf.abs(d_c_d), axis=2)\
                                    ),\
                             axis=1), \
                        tf.int32)
        
            with tf.variable_scope('coattentionencoder'):
                u_lstm_forward = tf.contrib.rnn.BasicLSTMCell(self.key_vec_size) 
                u_lstm_backward = tf.contrib.rnn.BasicLSTMCell(self.key_vec_size)
                big_u,_ = tf.nn.bidirectional_dynamic_rnn(cell_bw=u_lstm_backward,cell_fw=u_lstm_forward,dtype=tf.float32,inputs=d_c_d,time_major=False,sequence_length=d_c_d_length)
            
            # Dropout on the concatinated big U containing forward and backward pass
            output = tf.nn.dropout(tf.concat(big_u,2), self.keep_prob)


            return output

class CharacterLevelCNN(object):
    """
    Module that takes character embeddings for words in batch 
    and returns word level hidden representations 
    """

    def __init__(self, keep_prob, char_embedding_size, kernel_size=5, filters=100):
        self.keep_prob = keep_prob
        self.kernel_size = kernel_size
        self.filters = filters
        self.char_embedding_size = char_embedding_size

    def build_graph(self, char_embeddings):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          char_embeddings: Tensor shape (batch_size, phrase_len, word_len, char_embedding_size)

        Outputs:
          cnn_char_embeddings: Tensor shape (batch_size, phrase_len, filters)
        """
        with vs.variable_scope("CharLevelCNN"):
            batch_size = tf.shape(char_embeddings)[0]
            phrase_len = tf.shape(char_embeddings)[1]
            word_len = tf.shape(char_embeddings)[2]
            char_embedding_size = tf.shape(char_embeddings)[3]
            # b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

            # flatten
            # char_embeddings = tf.reshape(char_embeddings,[-1, word_len, char_embedding_size])
            char_embeddings = tf.reshape(char_embeddings, shape = [batch_size*phrase_len, word_len, self.char_embedding_size])

            conv = tf.layers.conv1d(inputs = char_embeddings, filters = self.filters, kernel_size = self.kernel_size, activation = tf.nn.relu, reuse = tf.AUTO_REUSE)  # shape (batch_size, phrase_len, word_len, filters)

            # unflatten
            conv = tf.reshape(conv, [batch_size, phrase_len, -1, self.filters])
            
            # h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Max-pooling over the outputs
            # cnn_char_embeddings = tf.nn.max_pool(conv, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
            
            cnn_char_embeddings = tf.reduce_max(conv, axis = 2)

            # dropout
            cnn_char_embeddings = tf.nn.dropout(cnn_char_embeddings, self.keep_prob)
            return cnn_char_embeddings


def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist

def get_shape(tensor):
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]
  return dims
