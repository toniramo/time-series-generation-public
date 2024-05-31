"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

predictive_metrics.py

Note: Use post-hoc RNN to classify original data and synthetic data

Output: discriminative score (np.abs(classification accuracy - 0.5))
"""

# Necessary Packages
#import tensorflow as tf
import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import numpy as np
from sklearn.metrics import accuracy_score
from lib.ext.utils import train_test_divide, extract_time, batch_generator

from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.util import nest

""" 
def _linear(args,
            output_size,
            bias,
            bias_initializer=None,
            kernel_initializer=None):
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

class GRUCell(RNNCell):
 # Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None):
    super(GRUCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def call(self, inputs, state):
    with vs.variable_scope("gates"):  # Reset gate and update gate.
      # We start with bias of 1.0 to not reset and not update.
      bias_ones = self._bias_initializer
      if self._bias_initializer is None:
        dtype = [a.dtype for a in [inputs, state]][0]
        bias_ones = init_ops.constant_initializer(1.0, dtype=dtype)
      value = math_ops.sigmoid(
          _linear([inputs, state], 2 * self._num_units, True, bias_ones,
                  self._kernel_initializer))
      r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
    with vs.variable_scope("candidate"):
      c = self._activation(
          _linear([inputs, r * state], self._num_units, True,
                  self._bias_initializer, self._kernel_initializer))
    new_h = u * state + (1 - u) * c
    return new_h, new_h


_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h")) """


def discriminative_score_metrics (ori_data, generated_data):
  """Use post-hoc RNN to classify original data and synthetic data
  
  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    
  Returns:
    - discriminative_score: np.abs(classification accuracy - 0.5)
  """
  # Initialization on the Graph
  tf.reset_default_graph()

  # Basic Parameters
  no, seq_len, dim = np.asarray(ori_data).shape    
    
  # Set maximum sequence length and each sequence length
  ori_time, ori_max_seq_len = extract_time(ori_data)
  generated_time, generated_max_seq_len = extract_time(ori_data)
  max_seq_len = max([ori_max_seq_len, generated_max_seq_len])  
     
  ## Builde a post-hoc RNN discriminator network 
  # Network parameters
  hidden_dim = int(dim/2)
  iterations = 2000
  batch_size = 128
    
  # Input place holders
  # Feature
  X = tf.placeholder(tf.float32, [None, max_seq_len, dim], name = "myinput_x")
  X_hat = tf.placeholder(tf.float32, [None, max_seq_len, dim], name = "myinput_x_hat")
    
  T = tf.placeholder(tf.int32, [None], name = "myinput_t")
  T_hat = tf.placeholder(tf.int32, [None], name = "myinput_t_hat")
    
  # discriminator function
  def discriminator (x, t):
    """Simple discriminator function.
    
    Args:
      - x: time-series data
      - t: time information
      
    Returns:
      - y_hat_logit: logits of the discriminator output
      - y_hat: discriminator output
      - d_vars: discriminator variables
    """
    with tf.variable_scope("discriminator", reuse = tf.AUTO_REUSE) as vs:
      #d_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'd_cell')
      d_cell = tf.contrib.rnn.GRUCell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'd_cell')
      d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, x, dtype=tf.float32, sequence_length = t)
      #y_hat_logit = tf.contrib.layers.fully_connected(d_last_states, 1, activation_fn=None)
      y_hat_logit = tf.layers.dense(d_last_states, 1, activation=None) 
      y_hat = tf.nn.sigmoid(y_hat_logit)
      d_vars = [v for v in tf.all_variables() if v.name.startswith(vs.name)]
    
    return y_hat_logit, y_hat, d_vars
    
  y_logit_real, y_pred_real, d_vars = discriminator(X, T)
  y_logit_fake, y_pred_fake, _ = discriminator(X_hat, T_hat)
        
  # Loss for the discriminator
  d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_logit_real, 
                                                                       labels = tf.ones_like(y_logit_real)))
  d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_logit_fake, 
                                                                       labels = tf.zeros_like(y_logit_fake)))
  d_loss = d_loss_real + d_loss_fake
    
  # optimizer
  d_solver = tf.train.AdamOptimizer().minimize(d_loss, var_list = d_vars)
        
  ## Train the discriminator   
  # Start session and initialize
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
    
  # Train/test division for both original and generated data
  train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
  train_test_divide(ori_data, generated_data, ori_time, generated_time)
    
  # Training step
  for itt in range(iterations):
          
    # Batch setting
    X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
    X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)
          
    # Train discriminator
    _, step_d_loss = sess.run([d_solver, d_loss], 
                              feed_dict={X: X_mb, T: T_mb, X_hat: X_hat_mb, T_hat: T_hat_mb})            
    
  ## Test the performance on the testing set    
  y_pred_real_curr, y_pred_fake_curr = sess.run([y_pred_real, y_pred_fake], 
                                                feed_dict={X: test_x, T: test_t, X_hat: test_x_hat, T_hat: test_t_hat})
    
  y_pred_final = np.squeeze(np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis = 0))
  y_label_final = np.concatenate((np.ones([len(y_pred_real_curr),]), np.zeros([len(y_pred_fake_curr),])), axis = 0)
    
  # Compute the accuracy
  acc = accuracy_score(y_label_final, (y_pred_final>0.5))
  discriminative_score = np.abs(0.5-acc)
    
  return discriminative_score  
