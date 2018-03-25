''' Useless code snippets for testing'''
# import os
# from tqdm import tqdm

# _PAD = b"<pad>"
# _UNK = b"<unk>"
# _START_VOCAB = [_PAD, _UNK]
# PAD_ID = 0
# UNK_ID = 1

# MAIN_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # relative path of the main directory
# DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "data") # relative path of data dir
# EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "experiments") # relative path of experiments dir

# train_context_path = os.path.join(DEFAULT_DATA_DIR, "train.context")
# train_qn_path = os.path.join(DEFAULT_DATA_DIR, "train.question")
# dev_context_path = os.path.join(DEFAULT_DATA_DIR, "dev.context")
# dev_qn_path = os.path.join(DEFAULT_DATA_DIR, "dev.question")


# def get_char_dict(*args):
#     print "Creating Character Mappings from training and dev sets..."
#     txt = ""
#     max_word_len = 0
#     for arg in args:
#         text_file = open(arg)
#         # context_line, qn_line, ans_line = text_file.readline()
#         for line in text_file:
#             txt += line
#             words = line.split()
#             max_len = max([len(w) for w in words])
#             if max_len>max_word_len: max_word_len = max_len
#     # Drop low frequency chars
#     char_frequency_dict = char_frequency(txt)
#     print char_frequency_dict
#     chars = list(set(txt))
#     chars.sort()
#     chars = _START_VOCAB + chars
#     char_vocab_size = len(chars)
#     char2id = dict((c, i) for i, c in enumerate(chars))
#     id2char = dict((i, c) for i, c in enumerate(chars))
#     print('Total number of unique chars: {}. Maximum Length: {}'.format(char_vocab_size, max_word_len))
#     # print chars
#     # print(char2id)
#     # print(id2char)
#     return char2id, id2char, char_vocab_size, max_word_len

# def char_frequency(str1):
#     dict = {}
#     for n in tqdm(str1):
#         keys = dict.keys()
#         if n in keys:
#             dict[n] += 1
#         else:
#             dict[n] = 1
#     return dict

# # char2id, id2char, char_vocab_size , _ = get_char_dict(train_context_path, train_qn_path, dev_context_path, dev_qn_path)
# char2id, id2char, char_vocab_size , _ = get_char_dict(dev_context_path)


# import sys
# import numpy as np
# import tensorflow as tf

# from TDNN import TDNN
# from base import Model

# from utils import progress
# from batch_loader import BatchLoader
# from ops import conv2d, batch_norm, highway

# class LSTMTDNN(Model):
#   """
#   Time-delayed Neural Network (cf. http://arxiv.org/abs/1508.06615v4)
#   """
#   def __init__(self, sess,
#                batch_size=100, rnn_size=650, layer_depth=2,
#                word_embed_dim=650, char_embed_dim=15,
#                feature_maps=[50, 100, 150, 200, 200, 200, 200],
#                kernels=[1,2,3,4,5,6,7], seq_length=35, max_word_length=65,
#                use_word=False, use_char=True, hsm=0, max_grad_norm=5,
#                highway_layers=2, dropout_prob=0.5, use_batch_norm=True,
#                checkpoint_dir="checkpoint", forward_only=False,
#                data_dir="data", dataset_name="pdb", use_progressbar=False):
#     """
#     Initialize the parameters for LSTM TDNN
#     Args:
#       rnn_size: the dimensionality of hidden layers
#       layer_depth: # of depth in LSTM layers
#       batch_size: size of batch per epoch
#       word_embed_dim: the dimensionality of word embeddings
#       char_embed_dim: the dimensionality of character embeddings
#       feature_maps: list of feature maps (for each kernel width)
#       kernels: list of kernel widths
#       seq_length: max length of a word
#       use_word: whether to use word embeddings or not
#       use_char: whether to use character embeddings or not
#       highway_layers: # of highway layers to use
#       dropout_prob: the probability of dropout
#       use_batch_norm: whether to use batch normalization or not
#       hsm: whether to use hierarchical softmax
#     """
#     self.sess = sess

#     self.batch_size = batch_size
#     self.seq_length = seq_length

#     # RNN
#     self.rnn_size = rnn_size
#     self.layer_depth = layer_depth

#     # CNN
#     self.use_word = use_word
#     self.use_char = use_char
#     self.word_embed_dim = word_embed_dim
#     self.char_embed_dim = char_embed_dim
#     self.feature_maps = feature_maps
#     self.kernels = kernels

#     # General
#     self.highway_layers = highway_layers
#     self.dropout_prob = dropout_prob
#     self.use_batch_norm = use_batch_norm

#     # Training
#     self.max_grad_norm = max_grad_norm
#     self.max_word_length = max_word_length
#     self.hsm = hsm

#     self.data_dir = data_dir
#     self.dataset_name = dataset_name
#     self.checkpoint_dir = checkpoint_dir

#     self.forward_only = forward_only
#     self.use_progressbar = use_progressbar

#     self.loader = BatchLoader(self.data_dir, self.dataset_name, self.batch_size, self.seq_length, self.max_word_length)
#     print('Word vocab size: %d, Char vocab size: %d, Max word length (incl. padding): %d' % \
#         (len(self.loader.idx2word), len(self.loader.idx2char), self.loader.max_word_length))

#     self.max_word_length = self.loader.max_word_length
#     self.char_vocab_size = len(self.loader.idx2char)
#     self.word_vocab_size = len(self.loader.idx2word)

#     # build LSTMTDNN model
#     self.prepare_model()

#     # load checkpoints
#     if self.forward_only == True:
#       if self.load(self.checkpoint_dir, self.dataset_name):
#         print("[*] SUCCESS to load model for %s." % self.dataset_name)
#       else:
#         print("[!] Failed to load model for %s." % self.dataset_name)
#         sys.exit(1)

#   def prepare_model(self):
#     with tf.variable_scope("LSTMTDNN"):
#       self.char_inputs = []
#       self.word_inputs = []
#       self.cnn_outputs = []

#       if self.use_char:
#         char_W = tf.get_variable("char_embed",
#             [self.char_vocab_size, self.char_embed_dim])
#       if self.use_word:
#         word_W = tf.get_variable("word_embed",
#             [self.word_vocab_size, self.word_embed_dim])

#       with tf.variable_scope("CNN") as scope:
#         self.char_inputs = tf.placeholder(tf.int32, [self.batch_size, self.seq_length, self.max_word_length])
#         self.word_inputs = tf.placeholder(tf.int32, [self.batch_size, self.seq_length])

#         char_indices = tf.split(1, self.seq_length, self.char_inputs)
#         word_indices = tf.split(1, self.seq_length, tf.expand_dims(self.word_inputs, -1))

#         for idx in xrange(self.seq_length):
#           char_index = tf.reshape(char_indices[idx], [-1, self.max_word_length])
#           word_index = tf.reshape(word_indices[idx], [-1, 1])

#           if idx != 0:
#             scope.reuse_variables()

#           if self.use_char:
#             # [batch_size x word_max_length, char_embed]
#             char_embed = tf.nn.embedding_lookup(char_W, char_index)

#             char_cnn = TDNN(char_embed, self.char_embed_dim, self.feature_maps, self.kernels)

#             if self.use_word:
#               word_embed = tf.nn.embedding_lookup(word_W, word_index)
#               cnn_output = tf.concat(1, [char_cnn.output, tf.squeeze(word_embed, [1])])
#             else:
#               cnn_output = char_cnn.output
#           else:
#             cnn_output = tf.squeeze(tf.nn.embedding_lookup(word_W, word_index))

#           if self.use_batch_norm:
#             bn = batch_norm()
#             norm_output = bn(tf.expand_dims(tf.expand_dims(cnn_output, 1), 1))
#             cnn_output = tf.squeeze(norm_output)

#           if highway:
#             #cnn_output = highway(input_, input_dim_length, self.highway_layers, 0)
#             cnn_output = highway(cnn_output, cnn_output.get_shape()[1], self.highway_layers, 0)

#           self.cnn_outputs.append(cnn_output)

#       with tf.variable_scope("LSTM") as scope:
#         self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size)
#         self.stacked_cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * self.layer_depth)

#         outputs, _ = tf.nn.rnn(self.stacked_cell,
#                                self.cnn_outputs,
#                                dtype=tf.float32)

#         self.lstm_outputs = []
#         self.true_outputs = tf.placeholder(tf.int64,
#             [self.batch_size, self.seq_length])

#         loss = 0
#         true_outputs = tf.split(1, self.seq_length, self.true_outputs)

#         for idx, (top_h, true_output) in enumerate(zip(outputs, true_outputs)):
#           if self.dropout_prob > 0:
#             top_h = tf.nn.dropout(top_h, self.dropout_prob)

#           if self.hsm > 0:
#             self.lstm_outputs.append(top_h)
#           else:
#             if idx != 0:
#               scope.reuse_variables()
#             proj = tf.nn.rnn_cell._linear(top_h, self.word_vocab_size, 0)
#             self.lstm_outputs.append(proj)

#           loss += tf.nn.sparse_softmax_cross_entropy_with_logits(self.lstm_outputs[idx], tf.squeeze(true_output))

#         self.loss = tf.reduce_mean(loss) / self.seq_length

#         tf.scalar_summary("loss", self.loss)
#         tf.scalar_summary("perplexity", tf.exp(self.loss))

#   def train(self, epoch):
#     cost = 0
#     target = np.zeros([self.batch_size, self.seq_length]) 

#     N = self.loader.sizes[0]
#     for idx in xrange(N):
#       target.fill(0)
#       x, y, x_char = self.loader.next_batch(0)
#       for b in xrange(self.batch_size):
#         for t, w in enumerate(y[b]):
#           target[b][t] = w

#       feed_dict = {
#           self.word_inputs: x,
#           self.char_inputs: x_char,
#           self.true_outputs: target,
#       }

#       _, loss, step, summary_str = self.sess.run(
#           [self.optim, self.loss, self.global_step, self.merged_summary], feed_dict=feed_dict)

#       self.writer.add_summary(summary_str, step)

#       if idx % 50 == 0:
#         if self.use_progressbar:
#           progress(idx/N, "epoch: [%2d] [%4d/%4d] loss: %2.6f" % (epoch, idx, N, loss))
#         else:
#           print("epoch: [%2d] [%4d/%4d] loss: %2.6f" % (epoch, idx, N, loss))

#       cost += loss
#     return cost / N

#   def test(self, split_idx, max_batches=None):
#     if split_idx == 1:
#       set_name = 'Valid'
#     else:
#       set_name = 'Test'

#     N = self.loader.sizes[split_idx]
#     if max_batches != None:
#       N = min(max_batches, N)

#     self.loader.reset_batch_pointer(split_idx)
#     target = np.zeros([self.batch_size, self.seq_length]) 

#     cost = 0
#     for idx in xrange(N):
#       target.fill(0)

#       x, y, x_char = self.loader.next_batch(split_idx)
#       for b in xrange(self.batch_size):
#         for t, w in enumerate(y[b]):
#           target[b][t] = w

#       feed_dict = {
#           self.word_inputs: x,
#           self.char_inputs: x_char,
#           self.true_outputs: target,
#       }

#       loss = self.sess.run(self.loss, feed_dict=feed_dict)

#       if idx % 50 == 0:
#         if self.use_progressbar:
#           progress(idx/N, "> %s: loss: %2.6f, perplexity: %2.6f" % (set_name, loss, np.exp(loss)))
#         else:
#           print(" > %s: loss: %2.6f, perplexity: %2.6f" % (set_name, loss, np.exp(loss)))

#       cost += loss

#     cost = cost / N
#     return cost

#   def run(self, epoch=25, 
#           learning_rate=1, learning_rate_decay=0.5):
#     self.current_lr = learning_rate

#     self.lr = tf.Variable(learning_rate, trainable=False)
#     self.opt = tf.train.GradientDescentOptimizer(self.lr)
#     #self.opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(self.loss)

#     # clip gradients
#     params = tf.trainable_variables()
#     grads = []
#     for grad in tf.gradients(self.loss, params):
#       if grad is not None:
#         grads.append(tf.clip_by_norm(grad, self.max_grad_norm))
#       else:
#         grads.append(grad)

#     self.global_step = tf.Variable(0, name="global_step", trainable=False)
#     self.optim = self.opt.apply_gradients(zip(grads, params),
#                                           global_step=self.global_step)

#     # ready for train
#     tf.initialize_all_variables().run()

#     if self.load(self.checkpoint_dir, self.dataset_name):
#       print("[*] SUCCESS to load model for %s." % self.dataset_name)
#     else:
#       print("[!] Failed to load model for %s." % self.dataset_name)

#     self.saver = tf.train.Saver()
#     self.merged_summary = tf.merge_all_summaries()
#     self.writer = tf.train.SummaryWriter("./logs", self.sess.graph_def)

#     self.log_loss = []
#     self.log_perp = []

#     if not self.forward_only:
#       for idx in xrange(epoch):
#         train_loss = self.train(idx)
#         valid_loss = self.test(1)

#         # Logging
#         self.log_loss.append([train_loss, valid_loss])
#         self.log_perp.append([np.exp(train_loss), np.exp(valid_loss)])

#         state = {
#           'perplexity': np.exp(train_loss),
#           'epoch': idx,
#           'learning_rate': self.current_lr,
#           'valid_perplexity': np.exp(valid_loss)
#         }
#         print(state)

#         # Learning rate annealing
#         if len(self.log_loss) > 1 and self.log_loss[idx][1] > self.log_loss[idx-1][1] * 0.9999:
#           self.current_lr = self.current_lr * learning_rate_decay
#           self.lr.assign(self.current_lr).eval()
#         if self.current_lr < 1e-5: break

#         if idx % 2 == 0:
#           self.save(self.checkpoint_dir, self.dataset_name)

#     test_loss = self.test(2)
#     print("[*] Test loss: %2.6f, perplexity: %2.6f" % (test_loss, np.exp(test_loss)))


# import tensorflow as tf
# import numpy as np

# keys =  tf.constant([[1,2,3], [4,5,6], [7,8,9], [10,11,12], ])

# values =  tf.constant([[13,0,1], [1,22,0], [0,0,7], [-1,-2,-3], [1,0,0],])

# weight = tf.constant([[1],[1],[1]])

# num_keys = tf.shape(keys)[0]
# num_values = tf.shape(values)[0]

# key_tile1 = tf.tile(keys, [1, num_values])
# key_tile1 = tf.reshape(key_tile1, [num_keys*num_values, 3])

# key_tile2 = tf.tile(values,[num_keys, 1])
# # key_tile2 = tf.reshape(key_tile2, [num_keys*num_values, 3])

# mixedtile = tf.multiply(key_tile1, key_tile2)

# finalmatrix = tf.matmul(mixedtile, weight)

# final = tf.reshape(finalmatrix, [num_keys, num_values])

# with tf.Session() as sess:
#     print sess.run(mixedtile)    
#     print sess.run(final)


# tf.set_random_seed(50)
# w =  tf.random_normal(shape = [5, 1])
# t = tf.expand_dims(w, 0)

# s = [w,t]

# with tf.Session() as sess:
#     print sess.run(s)

# import sys; sys.exit()


# batch_size = 3   # batch_size
# num_values = 2   # question_len
# num_keys   = 4   # context_len
# key_vec_size = 5
# value_vec_size = 5

# tf.set_random_seed(50)
# values =  tf.random_normal(shape=[batch_size, num_values, value_vec_size])
# keys   =  tf.random_normal(shape=[batch_size, num_keys, key_vec_size])

# values_t = tf.transpose(values, perm=[0, 2, 1])

# weights_sim1 =  tf.random_normal(shape = [key_vec_size, 1])
# weights_sim2 =  tf.random_normal(shape = [value_vec_size, 1])
# weights_sim3 =  tf.random_normal(shape = [key_vec_size])

# S1 = tf.reshape(tf.matmul(tf.reshape(keys, [-1, key_vec_size]), weights_sim1),[-1, num_keys, 1]) # shape : (batch_size, num_keys, 1)
# S2 = tf.reshape(tf.matmul(tf.reshape(values, [-1, value_vec_size]), weights_sim2),[-1, num_values, 1]) # shape : (batch_size, num_values, 1)
# S2 = tf.transpose(S2, perm=[0, 2, 1])   # shape : (batch_size, 1,   num_values)

# # GPU efficient version
# weights_sim3 = tf.expand_dims(tf.expand_dims(weights_sim3,0),1)  # make it (1, 1, key_vec_size)
# ctile = tf.multiply(keys, weights_sim3) #(batch_size, num_keys, value_vec_size)
# S3 = tf.matmul(ctile, values_t)

# S=S1+S2

# # weights_sim3 =  tf.expand_dims(weights_sim3,1)
# weights_sim = tf.concat([weights_sim1, weights_sim2], axis=0)


# Ctile = tf.tile(keys, [1, 1, num_values])
# Ctile = tf.reshape(Ctile, [-1,num_values*num_keys, key_vec_size])
# Qtile = tf.tile(values,[1, num_keys, 1])

# xxx = tf.concat([Ctile, Qtile], axis=2)


# Sxxx = tf.reshape(tf.matmul(tf.reshape(xxx, [-1, key_vec_size+value_vec_size]), weights_sim),[-1, num_keys, num_values])

# with tf.Session() as sess:
#     print np.array(sess.run([S, Sxxx]))


# keys_mask = tf.constant([[1, 1, 0, 0], [1, 1, 1, 0]])
# values_mask = tf.constant([[1, 0], [1, 1]])

# sim_mask1 = tf.expand_dims(keys_mask, 2)*tf.expand_dims(values_mask, 1)
# # sim_mask2 = tf.reshape(keys_mask, [-1, num_keys, 1])+tf.reshape(values_mask, [-1, 1, num_values])

# with tf.Session() as sess:
#     print sess.run(sim_mask1)
#     # print sess.run(sim_mask2)



# This doesn't work for GPU:  run out of memory
# Ctile = tf.tile(keys, [1, 1, num_values])
# Ctile = tf.reshape(Ctile, [-1,num_values*num_keys, self.key_vec_size])
# Qtile = tf.tile(values,[1, num_keys, 1])            
# CQtile = tf.multiply(Ctile, Qtile)   # shape: (batchsize, num_keys*num_values, vec_size)

# S3 = tf.reshape(tf.matmul(tf.reshape(CQtile, [-1, self.value_vec_size]), weights_sim3),[-1, num_values*num_keys, 1])
# S3 = tf.reshape(S3, [-1, num_keys, num_values])



# S3 = tf.reshape(tf.matmul(tf.reshape(CQtile, [-1, self.value_vec_size]), weights_sim3),[-1, num_values*num_keys, 1])
# S3 = tf.reshape(S3, [-1, num_keys, num_values])


# Ctile = tf.tile(keys, [1, 1, num_values])
# Ctile = tf.reshape(Ctile, [-1,num_values*num_keys, self.key_vec_size])
# Qtile = tf.tile(values,[1, num_keys, 1])            
# CQtile = tf.multiply(Ctile, Qtile)   # shape: (batchsize, num_keys*num_values, vec_size)

