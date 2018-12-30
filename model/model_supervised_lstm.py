import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
import keras.preprocessing.sequence as pp

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)


def get_transition_matrix(docnade_indices, lstm_indices, docnade_seq_lengths, rnn_seq_lengths, mapping_dict):
    assert (docnade_indices.shape[0] == lstm_indices.shape[0])

    batch_size, d_seq_len = docnade_indices.shape
    l_seq_len = lstm_indices.shape[1]

    #transition_matrix = np.zeros((batch_size, d_seq_len, l_seq_len), dtype=np.float32)
    transition_matrices = []
    for d_indices, l_indices, d_seq, l_seq in zip(docnade_indices, lstm_indices, docnade_seq_lengths, rnn_seq_lengths):
        mapped_d_indices = np.zeros((d_seq, 1))
        mapped_d_indices[:,0] = np.array([int(mapping_dict[int(d_index)]) for d_index in d_indices[:d_seq]])
        temp_d = np.tile(mapped_d_indices, (1, l_seq))

        mapped_l_indices = np.zeros((1, l_seq))
        mapped_l_indices[0, :] = l_indices[:l_seq]
        temp_l = np.tile(mapped_l_indices, (d_seq, 1))

        transition_matrix = np.zeros((d_seq_len, l_seq_len), dtype=np.float32)
        transition_matrix[:d_seq, :l_seq] = np.array((temp_d == temp_l), dtype=np.float32)
        transition_matrices.append(transition_matrix)
    transition_matrices_stacked = np.stack(transition_matrices)

    return transition_matrices_stacked

def get_transition_indices(docnade_indices, lstm_indices, docnade_seq_lengths, rnn_seq_lengths, mapping_dict, max_seq_len):
    # print('docnade_indices.shape[0]:', docnade_indices.shape[0])
    # print('lstm_indices.shape[0]:', lstm_indices.shape[0])
    assert (docnade_indices.shape[0] == lstm_indices.shape[0])

    batch_size, d_seq_len = docnade_indices.shape
    l_seq_len = lstm_indices.shape[1]

    transition_indices = []
    max_length = 0
    for d_indices, l_indices, d_seq, l_seq in zip(docnade_indices, lstm_indices, docnade_seq_lengths, rnn_seq_lengths):
        #mapped_l_indices = np.zeros((1, l_seq))
        mapped_l_indices = np.array(l_indices[:l_seq])

        d_transition_indices = []
        for d_index in d_indices:
            indices = np.where(mapped_l_indices == mapping_dict[d_index])[0].tolist()
            if indices == []:
                d_transition_indices.append([0])
            else:
                d_transition_indices.append(indices)
        d_transition_indices = pp.pad_sequences(d_transition_indices, dtype='int32', padding='post', value=max_seq_len)
        transition_indices.append(d_transition_indices)
        if d_transition_indices.shape[1] > max_length:
            max_length = d_transition_indices.shape[1]

    transition_index_matrix = np.ones((batch_size, d_seq_len, max_length), dtype=np.int32) * max_seq_len
    for i, mat in enumerate(transition_indices):
        transition_index_matrix[i, :, :mat.shape[1]] = mat

    #transition_indices = pp.pad_sequences(transition_indices, dtype='int32', padding='post', value=max_seq_len)
    #transition_index_matrix = np.array(transition_index_matrix)
    return transition_index_matrix


def vectors_docnade(model, data, data_lstm, mapping_dict, session, max_seq_len):
    vecs = []
    for _, x, seq_lengths in data:
        _, x_rnn, rnn_seq_lengths = next(data_lstm)
        #transition_matrix = get_transition_matrix(x, x_rnn, seq_lengths, rnn_seq_lengths, mapping_dict)
        transition_matrix = get_transition_indices(x, x_rnn, seq_lengths, rnn_seq_lengths, mapping_dict, max_seq_len)
        vecs.extend(
            session.run([model.h], feed_dict={
                model.x: x,
                model.seq_lengths: seq_lengths,
                model.x_rnn: x_rnn,
                model.rnn_seq_lengths: rnn_seq_lengths,
                model.rnn_transition_matrix: transition_matrix
            })[0]
        )
    return np.array(vecs)

def vectors_comb(model, data, data_lstm, mapping_dict, combination_type, session, max_seq_len, docnade_loss_weight, lstm_loss_weight):
    vecs = []
    for _, x, seq_lengths in data:
        _, x_rnn, rnn_seq_lengths = next(data_lstm)
        #transition_matrix = get_transition_matrix(x, x_rnn, seq_lengths, rnn_seq_lengths, mapping_dict)
        transition_matrix = get_transition_indices(x, x_rnn, seq_lengths, rnn_seq_lengths, mapping_dict, max_seq_len)
        if combination_type == 'concat':
            vecs.extend(
                session.run([model.h_comb_concat], feed_dict={
                    model.x: x,
                    model.seq_lengths: seq_lengths,
                    model.x_rnn: x_rnn,
                    model.rnn_seq_lengths: rnn_seq_lengths,
                    model.rnn_transition_matrix: transition_matrix,
                    model.docnade_loss_weight: docnade_loss_weight,
                    model.lstm_loss_weight: lstm_loss_weight
                })[0]
            )
        elif combination_type == 'sum':
            vecs.extend(
                session.run([model.h_comb_sum], feed_dict={
                    model.x: x,
                    model.seq_lengths: seq_lengths,
                    model.x_rnn: x_rnn,
                    model.rnn_seq_lengths: rnn_seq_lengths,
                    model.rnn_transition_matrix: transition_matrix,
                    model.docnade_loss_weight: docnade_loss_weight,
                    model.lstm_loss_weight: lstm_loss_weight
                })[0]
            )
        elif combination_type == 'projected':
            vecs.extend(
                session.run([model.h_comb_last], feed_dict={
                    model.x: x,
                    model.seq_lengths: seq_lengths,
                    model.x_rnn: x_rnn,
                    model.rnn_seq_lengths: rnn_seq_lengths,
                    model.rnn_transition_matrix: transition_matrix,
                    model.docnade_loss_weight: docnade_loss_weight,
                    model.lstm_loss_weight: lstm_loss_weight
                })[0]
            )
        
    return np.array(vecs)

def vectors_comb_common_space(model, data, data_lstm, mapping_dict, session, max_seq_len):
    vecs = []
    for _, x, seq_lengths in data:
        _, x_rnn, rnn_seq_lengths = next(data_lstm)
        #transition_matrix = get_transition_matrix(x, x_rnn, seq_lengths, rnn_seq_lengths, mapping_dict)
        transition_matrix = get_transition_indices(x, x_rnn, seq_lengths, rnn_seq_lengths, mapping_dict, max_seq_len)
        vecs.extend(
            session.run([model.h_comb_last], feed_dict={
                model.x: x,
                model.seq_lengths: seq_lengths,
                model.x_rnn: x_rnn,
                model.rnn_seq_lengths: rnn_seq_lengths,
                model.rnn_transition_matrix: transition_matrix
            })[0]
        )
    return np.array(vecs)

def vectors_lstm(model, data, data_lstm, mapping_dict, session, max_seq_len):
    vecs = []
    for _, x, seq_lengths in data:
        _, x_rnn, rnn_seq_lengths = next(data_lstm)
        #transition_matrix = get_transition_matrix(x, x_rnn, seq_lengths, rnn_seq_lengths, mapping_dict)
        transition_matrix = get_transition_indices(x, x_rnn, seq_lengths, rnn_seq_lengths, mapping_dict, max_seq_len)
        vecs.extend(
            session.run([model.last_lstm_h], feed_dict={
                model.x: x,
                model.seq_lengths: seq_lengths,
                model.x_rnn: x_rnn,
                model.rnn_seq_lengths: rnn_seq_lengths,
                model.rnn_transition_matrix: transition_matrix
            })[0]
        )
    return np.array(vecs)


def loss(model, data, session):
    loss = []
    for _, x, seq_lengths in data:
        loss.append(
            session.run([model.loss], feed_dict={
                model.x: x,
                model.seq_lengths: seq_lengths
            })[0]
        )
    return sum(loss) / len(loss)


def gradients(opt, loss, vars, step, max_gradient_norm=None, dont_clip=[]):
    gradients = opt.compute_gradients(loss, vars)
    if max_gradient_norm is not None:
        to_clip = [(g, v) for g, v in gradients if v.name not in dont_clip]
        not_clipped = [(g, v) for g, v in gradients if v.name in dont_clip]
        gradients, variables = zip(*to_clip)
        clipped_gradients, _ = clip_ops.clip_by_global_norm(
            gradients,
            max_gradient_norm
        )
        gradients = list(zip(clipped_gradients, variables)) + not_clipped

    # Add histograms for variables, gradients and gradient norms
    for gradient, variable in gradients:
        if isinstance(gradient, ops.IndexedSlices):
            grad_values = gradient.values
        else:
            grad_values = gradient
        if grad_values is None:
            print('warning: missing gradient: {}'.format(variable.name))
        if grad_values is not None:
            tf.summary.histogram(variable.name, variable)
            tf.summary.histogram(variable.name + '/gradients', grad_values)
            tf.summary.histogram(
                variable.name + '/gradient_norm',
                clip_ops.global_norm([grad_values])
            )

    return opt.apply_gradients(gradients, global_step=step)


def linear(input_comb, input_docnade, input_lstm, output_dim, scope=None, stddev=None, W_initializer=None):
    const = tf.constant_initializer(0.0)

    if W_initializer is None:
        if stddev:
            norm = tf.random_normal_initializer(stddev=stddev)
        else:
            norm = tf.random_normal_initializer(
                stddev=np.sqrt(2.0 / input_comb.get_shape()[1].value)
            )

        with tf.variable_scope(scope or 'linear'):
            w = tf.get_variable(
                'V',
                [input_comb.get_shape()[1], output_dim],
                initializer=norm
            )
    else:
        w = tf.get_variable(
            "V",
            # [params.hidden_size, params.vocab_size],
            initializer=tf.transpose(W_initializer)
        )
    
    b = tf.get_variable('b', [output_dim], initializer=const)

    comb_logits = tf.nn.xw_plus_b(input_comb, w, b, name='docnade_loss_comb_logits')
    docnade_logits = tf.nn.xw_plus_b(input_docnade, w, b, name='docnade_loss_docnade_logits')
    lstm_logits = tf.nn.xw_plus_b(input_lstm, w, b, name='docnade_loss_lstm_logits')

    return comb_logits, docnade_logits, lstm_logits

"""
def linear_reload(input_comb, input_docnade, input_lstm, output_dim, scope=None, stddev=None, V_reload=None, b_reload=None):
    w = tf.get_variable(
        "V",
        initializer=V_reload,
        trainable=False
    )
    
    b = tf.get_variable(
        'b',
        initializer=b_reload,
        trainable=False
    )

    comb_logits = tf.nn.xw_plus_b(input_comb, w, b)
    docnade_logits = tf.nn.xw_plus_b(input_docnade, w, b)
    lstm_logits = tf.nn.xw_plus_b(input_lstm, w, b)

    return comb_logits, docnade_logits, lstm_logits
"""

def masked_sequence_cross_entropy_loss(
    x,
    seq_lengths,
    logits,
    loss_function=None,
    norm_by_seq_lengths=True,
    name=None
):
    '''
    Compute the cross-entropy loss between all elements in x and logits.
    Masks out the loss for all positions greater than the sequence
    length (as we expect that sequences may be padded).

    Optionally, also either use a different loss function (eg: sampled
    softmax), and/or normalise the loss for each sequence by the
    sequence length.
    '''
    batch_size = tf.shape(x)[0]
    #batch_len = tf.cast(tf.shape(x)[1], dtype=tf.int64)
    batch_len = tf.shape(x)[1]
    labels = tf.reshape(x, [-1])

    max_doc_length = tf.reduce_max(seq_lengths)
    mask = tf.less(
        #tf.range(0, max_doc_length, 1),
        tf.range(0, batch_len, 1),
        tf.reshape(seq_lengths, [batch_size, 1])
    )
    mask = tf.reshape(mask, [-1])
    mask = tf.to_float(tf.where(
        mask,
        tf.ones_like(labels, dtype=tf.float32),
        tf.zeros_like(labels, dtype=tf.float32)
    ))

    if loss_function is None:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=labels
        )
    else:
        loss = loss_function(logits, labels)
    loss *= mask
    loss = tf.reshape(loss, [batch_size, -1])
    loss = tf.reduce_sum(loss, axis=1)
    loss_unnormed = loss
    if norm_by_seq_lengths:
        loss = loss / tf.to_float(seq_lengths)
    return tf.reduce_mean(loss, name=name + '_normed'), labels, mask, tf.reduce_mean(loss_unnormed, name=name + '_unnormed')

class BiRnnLstm(object):
    def __init__(self, input, rnn_seq_lengths, transition_matrix, hidden_size, vocab_size, batch_size, indices,
                 rnn_vocab_length, trainable_val, max_seq_len, include_lstm_loss, deep, deep_sizes=None, activation=None, W_initializer=None, W_docnade=None):
        self.input = input
        self.transition_matrix = transition_matrix

        # Do an embedding lookup for each word in each sequence
        
        with tf.device('/cpu:0'):
            # Initialisation scheme taken from the original DocNADE source
            if W_initializer is None:
                """
                max_embed_init = 1.0 / (rnn_vocab_length * hidden_size)
                W = tf.get_variable(
                    'embeddings_rnn',
                    [rnn_vocab_length, hidden_size],
                    initializer=tf.random_uniform_initializer(
                        maxval=max_embed_init
                    ),
                    trainable=trainable_val
                )
                """
                pass
            else:
                W = tf.get_variable(
                    'embeddings_rnn',
                    # [rnn_vocab_length, hidden_size],
                    initializer=W_initializer,
                    #trainable=trainable_val
                    trainable=False
                )

                self.embeddings = tf.nn.embedding_lookup(W, input, name='rnn_embedding_lookup')
        
        #W = W_initializer
        self.embeddings_docnade_prior = tf.nn.embedding_lookup(W_docnade, input, name='rnn_embedding_lookup_docnade_prior')

        if W_initializer is None:
            embedding_input = self.embeddings_docnade_prior
        else:
            embedding_input = self.embeddings + self.embeddings_docnade_prior
        """
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, timesteps, n_input)
        # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)
        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
        #max_seq_len = 59
        #new_input = tf.unstack(input, max_seq_len, 1)

        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = rnn.BasicLSTMCell(hidden_size, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = rnn.BasicLSTMCell(hidden_size, forget_bias=1.0)

        # Get lstm cell output
        #try:
        #    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, new_input,
        #                                        dtype=tf.float32)
        #except Exception: # Old TensorFlow version only returns outputs not states
        #    outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, new_input,
        #                                        dtype=tf.float32)

        #try:
        #    outputs, _, _ = rnn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, new_input,
        #                                        dtype=tf.float32)
        #except Exception: # Old TensorFlow version only returns outputs not states
        outputs_rnn, final_states_rnn = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, input,
                                            dtype=tf.float32)

        outputs_fw, outputs_bw = outputs_rnn
        final_states_fw, final_states_bw = final_states_rnn

        #outputs = tf.concat([
        #    outputs_fw, outputs_bw
        #], axis=2)

        self.outputs_fw = outputs_fw
        self.outputs_bw = outputs_bw

        #self.outputs = outputs
        #self.outputs_shape = tf.shape(outputs)
        #outputs_T = tf.transpose(outputs, [0,1,2])
        #self.outputs_T_shape = tf.shape(outputs_T)

        #hidden = tf.matmul(self.transition_matrix, outputs_T)
        hidden_fw = tf.matmul(self.transition_matrix, outputs_fw)
        hidden_bw = tf.matmul(self.transition_matrix, outputs_bw)
        hidden = tf.concat([
            hidden_fw, hidden_bw
        ], axis=2)
        self.hidden_shape = tf.shape(hidden)

        self.last_h_fw = tf.gather_nd(hidden_fw, indices)
        self.last_h_bw = hidden_bw[:, 0, :]

        #self.last_h = tf.concat([
        #    self.last_h_fw, self.last_h_bw
        #], axis=1)
        self.last_h = self.last_h_fw

        hidden_reshape = tf.reshape(hidden, [-1, 2*hidden_size])
        self.hidden_reshape_shape = tf.shape(hidden_reshape)

        with tf.device('/cpu:0'):
            max_embed_init = 1.0 / (2 * hidden_size * vocab_size)
            self.W = tf.get_variable(
                'out',
                [2*hidden_size, vocab_size],
                dtype=tf.float32,
                initializer=tf.random_uniform_initializer(
                    maxval=max_embed_init
                )
            )

        self.logits = tf.matmul(hidden_reshape, self.W)
        """
        #max_seq_len = 88
        #max_seq_len = 14812

        #def linear_activation(x):
        #    return x

        def lstm_cell(hidden_size, act=activation, name=None):
            #return rnn.BasicLSTMCell(hidden_size, name=name)
            return rnn.LSTMCell(hidden_size, activation=act, name=name)

        new_input = tf.unstack(embedding_input, max_seq_len, 1, name='lstm_unstack')
        #lstm_cell = rnn.LSTMCell(hidden_size, forget_bias=1.0, activation=linear_activation)

        if not deep:
            #lstm_cell = rnn.LSTMCell(hidden_size, forget_bias=1.0, name='lstm_cell')
            lstm_cell = rnn.LSTMCell(hidden_size, activation=activation, forget_bias=1.0, name='lstm_cell')
            outputs, state = rnn.static_rnn(lstm_cell, new_input, dtype=tf.float32)
            #self.weights = lstm_cell._kernel
            #self.biases  = lstm_cell._bias
        else:
            stacked_lstm_cell = rnn.MultiRNNCell([lstm_cell(deep_sizes[i], name='lstm_cell_' + str(i)) for i in range(len(deep_sizes))])
            outputs, state = rnn.static_rnn(stacked_lstm_cell, new_input, dtype=tf.float32)

        self.outputs = outputs
        self.outputs_shape = tf.shape(outputs)
        outputs_T = tf.transpose(outputs, [1,0,2])
        self.outputs_T_shape = tf.shape(outputs_T)

        self.last_h = tf.gather_nd(outputs_T, indices, name="last_hidden_lstm")

        outputs_T_new = tf.concat([
            outputs_T, tf.zeros([batch_size, 1, hidden_size])
        ], axis=1)
        self.outputs_T = outputs_T

        #hidden = tf.matmul(self.transition_matrix, outputs_T)
        elems = (outputs_T_new, self.transition_matrix)
        hidden = tf.map_fn(lambda inputs: tf.nn.embedding_lookup(inputs[0], inputs[1]), elems, dtype=tf.float32)
        #hidden = tf.nn.embedding_lookup(outputs_T, self.transition_matrix)[0]
        hidden = tf.reduce_sum(hidden, axis=2)

        #outputs_T = outputs_T[:, :-1, :]

        hidden = tf.concat([
            tf.zeros([batch_size, 1, hidden_size], dtype=tf.float32), hidden
        ], axis=1)
        hidden = hidden[:, :-1, :]

        self.hidden = hidden
        #self.hidden = tf.Variable(0.0, trainable=False, name='hidden_lstm')
        #tf.assign(self.hidden, hidden, name='lstm_hidden_assign')

        if include_lstm_loss:
            #self.hidden_reshape = tf.reshape(hidden, [-1, hidden_size])
            self.outputs_T_reshape = tf.reshape(outputs_T, [-1, hidden_size])

            with tf.device('/cpu:0'):
                max_embed_init = 1.0 / (hidden_size * rnn_vocab_length)
                self.V_rnn = tf.get_variable(
                    'V_rnn',
                    [hidden_size, rnn_vocab_length],
                    dtype=tf.float32,
                    initializer=tf.random_uniform_initializer(maxval=max_embed_init)
                )

                self.bias_rnn = tf.get_variable(
                    'd_rnn',
                    [rnn_vocab_length],
                    initializer=tf.constant_initializer(0)
                )

            #self.logits = tf.matmul(self.hidden_reshape, self.W)
            self.logits = tf.nn.xw_plus_b(self.outputs_T_reshape, self.V_rnn, self.bias_rnn, name='lstm_loss_lstm_logits')

            self.loss_normed, _, _, self.loss_unnormed  = masked_sequence_cross_entropy_loss(
                input,
                rnn_seq_lengths,
                self.logits,
                loss_function=None,
                norm_by_seq_lengths=True,
                name='lstm_loss'
            )


class MLP(object):
    def __init__(self, input, labels, num_classes=10, hidden_sizes=[]):
        self.input = input
        self.labels = labels
        self.num_layers = len(hidden_sizes)
        self.num_classes = num_classes
        self.hidden_sizes = hidden_sizes

        self.U_list = []
        self.d_list = []
        hidden_sizes.append(num_classes)

        for i in range(self.num_layers):
            max_U_init = 1.0 / (hidden_sizes[i] * hidden_sizes[i+1])
            U = tf.get_variable(
                'U_' + str(i),
                [hidden_sizes[i], hidden_sizes[i+1]],
                initializer=tf.random_uniform_initializer(
                    maxval=max_U_init
                )
            )
            d = tf.get_variable(
                'd_' + str(i),
                [hidden_sizes[i+1]],
                initializer=tf.constant_initializer(0)
            )
            self.U_list.append(U)
            self.d_list.append(d)

        # Forward pass
        #temp = tf.matmul(input, self.U_list[0]) + self.d_list[0]
        temp = tf.nn.xw_plus_b(input, self.U_list[0], self.d_list[0])
        for i in range(1, self.num_layers):
            #temp = tf.matmul(temp, self.U_list[i]) + self.d_list[i]
            temp = tf.nn.xw_plus_b(temp, self.U_list[i], self.d_list[i])
        disc_logits = temp

        one_hot_labels = tf.one_hot(labels, depth=num_classes)

        self.pred_labels = tf.argmax(disc_logits, axis=1)

        self.disc_loss = tf.losses.softmax_cross_entropy(
            onehot_labels=one_hot_labels,
            logits=disc_logits,
        )

        self.disc_accuracy = tf.metrics.accuracy(labels, self.pred_labels)
        self.disc_output = disc_logits


class DocNADE(object):
    def __init__(self, x, y, seq_lengths, params, x_rnn, rnn_seq_lengths, rnn_transition_matrix, rnn_vocab_length,
                 max_seq_len, docnade_loss_weight, lstm_loss_weight, lambda_hidden_lstm, W_initializer_docnade=None, W_initializer_rnn=None):
        print("\n\nDocNADE with RNN-LSTM\n\n")
        self.x = x
        self.y = y
        self.seq_lengths = seq_lengths
        self.max_seq_len = max_seq_len

        self.x_rnn = x_rnn
        self.rnn_seq_lengths = rnn_seq_lengths
        self.rnn_transition_matrix = rnn_transition_matrix

        self.docnade_loss_weight = docnade_loss_weight
        self.lstm_loss_weight = lstm_loss_weight

        self.lambda_hidden_lstm = lambda_hidden_lstm

        batch_size = tf.shape(x)[0]

        self.b_s_x = tf.shape(x)
        self.b_s_lstm = tf.shape(x_rnn)
        self.docnade_initializer_embeddings = W_initializer_docnade

        if params.common_space:
            with tf.device('/cpu:0'):
                # Initialisation scheme taken from the original DocNADE source
                max_embed_init = 1.0 / (params.hidden_size * params.hidden_size)
                W_proj_docnade = tf.get_variable(
                    'proj_docnade',
                    [params.hidden_size, params.hidden_size],
                    initializer=tf.random_uniform_initializer(
                        maxval=max_embed_init
                    ),
                    trainable=True
                )

                W_proj_lstm = tf.get_variable(
                    'proj_lstm',
                    [params.hidden_size, params.hidden_size],
                    initializer=tf.random_uniform_initializer(
                        maxval=max_embed_init
                    ),
                    trainable=True
                )

                bias_proj = tf.get_variable(
                    'bias_proj',
                    [params.hidden_size],
                    initializer=tf.constant_initializer(0),
                    trainable=True
                )

        # Do an embedding lookup for each word in each sequence
        with tf.device('/cpu:0'):
            # Initialisation scheme taken from the original DocNADE source
            if W_initializer_docnade is None:
                max_embed_init = 1.0 / (params.vocab_size * params.hidden_size)
                W = tf.get_variable(
                    'embeddings_docnade',
                    [params.vocab_size, params.hidden_size],
                    initializer=tf.random_uniform_initializer(
                        maxval=max_embed_init
                    ),
                    trainable=params.update_docnade_w
                )
            else:
                W = tf.get_variable(
                    'embeddings_docnade',
                    #[params.vocab_size, params.hidden_size],
                    initializer=W_initializer_docnade,
                    trainable=params.update_docnade_w
                )
            
            self.embeddings = tf.nn.embedding_lookup(W, x, name='docnade_embedding_lookup')

        bias = tf.get_variable(
            'bias_docnade',
            [params.hidden_size],
            initializer=tf.constant_initializer(0)
        )

        #############################################################################################
        ################################## Deep DocNADE Parameters ##################################
        W_list = []
        bias_list = []
        if params.deep:
            in_size = params.hidden_size
            for index, size in enumerate(params.deep_hidden_sizes):
                out_size = size
                max_embed_init = 1.0 / (in_size * out_size)
                W_temp = tf.get_variable(
                    'embedding_deep_' + str(index),
                    [in_size, out_size],
                    initializer=tf.random_uniform_initializer(
                        maxval=max_embed_init
                    )
                )
                bias_temp = tf.get_variable(
                    'bias_deep_' + str(index),
                    [out_size],
                    initializer=tf.constant_initializer(0)
                )
                W_list.append(W_temp)
                bias_list.append(bias_temp)
                in_size = out_size

        #############################################################################################

        self.embeddings_shape = tf.shape(self.embeddings)

        # Compute the hidden layer inputs: each gets summed embeddings of
        # previous words
        def sum_embeddings(previous, current):
            return previous + current

        h = tf.scan(sum_embeddings, tf.transpose(self.embeddings, [1, 2, 0]))
        self.h_shape = tf.shape(h)
        h = tf.transpose(h, [2, 0, 1])
        self.h_transpose_shape = tf.shape(h)

        #bias_lstm = tf.get_variable(
        #    'bias_lstm',
        #    [params.hidden_size],
        #    initializer=tf.constant_initializer(0)
        #)

        #self.bias_shape = tf.shape(bias)
        

        # add initial zero vector to each sequence, will then generate the
        # first element using just the bias term
        h = tf.concat([
            tf.zeros([batch_size, 1, params.hidden_size], dtype=tf.float32), h
        ], axis=1)
        
        self.pre_act = h
        #self.pre_act_shape = tf.shape(self.pre_act)

        ## Apply activation
        #if params.activation == 'sigmoid':
        #    h_docnade = tf.sigmoid(h + bias)
        #elif params.activation == 'tanh':
        #    h_docnade = tf.tanh(h + bias)
        #elif params.activation == 'relu':
        #    h_docnade = tf.nn.relu(h + bias)
        #else:
        #    print('Invalid value for activation: %s' % (params.activation))
        #    exit()
        #self.aft_act = h_docnade

        # Extract final state for each sequence in the batch
        indices = tf.stack([
            tf.range(batch_size),
            tf.to_int32(seq_lengths)
        ], axis=1)
        self.indices = indices
        self.h = tf.gather_nd(h, indices, name="last_hidden_docnade")

        #h_docnade = h_docnade[:, :-1, :]
        h = h[:, :-1, :]
        h = tf.reshape(h, [-1, params.hidden_size])

        ################ Deep network forward propagation ###################
        if params.deep:
            for i in range(len(params.deep_hidden_sizes)):
                h = tf.nn.xw_plus_b(h, W_list[i], bias_list[i])
                self.h = tf.nn.xw_plus_b(self.h, W_list[i], bias_list[i])
                if params.activation == 'sigmoid':
                    h = tf.sigmoid(h)
                    self.h = tf.sigmoid(self.h)
                elif params.activation == 'tanh':
                    h = tf.tanh(h)
                    self.h = tf.tanh(self.h)
                else:
                    h = tf.nn.relu(h)
                    self.h = tf.nn.relu(self.h)

        ####################### SUPERVISED NETWORK ##########################

        #self.final_state_h_shape = tf.shape(self.h)

        #########################################################################
        ######################## ADDING LSTM HIDDEN LAYER #######################

        lstm_indices = tf.stack([
            tf.range(batch_size),
            tf.to_int32(rnn_seq_lengths) - 1
        ], axis=1)
        self.lstm_indices = lstm_indices

        #input_rnn = tf.nn.embedding_lookup(glove_embeddings, x_rnn)

        if params.activation == 'sigmoid':
            actn = tf.sigmoid
        elif params.activation == 'tanh':
            actn = tf.tanh
        elif params.activation == 'relu':
            actn = tf.nn.relu
        else:
            print('Invalid value for activation: %s' % (params.activation))
            exit()

        #self.lstm = BiRnnLstm(x_rnn, rnn_seq_lengths, rnn_transition_matrix, params.hidden_size, params.vocab_size, batch_size, lstm_indices,
        #                      rnn_vocab_length, params.update_rnn_w, max_seq_len, params.include_lstm_loss, params.deep, deep_sizes=params.deep_hidden_sizes, 
        #                      activation=actn, W_initializer=W_initializer_rnn)
        self.lstm = BiRnnLstm(x_rnn, rnn_seq_lengths, rnn_transition_matrix, params.hidden_size, params.vocab_size, batch_size, lstm_indices,
                              rnn_vocab_length, params.update_rnn_w, max_seq_len, params.include_lstm_loss, params.deep, deep_sizes=params.deep_hidden_sizes, 
                              activation=actn, W_initializer=W_initializer_rnn, W_docnade=W)
        #self.lstm_logits = self.lstm.logits
        #self.lstm_logits_shape = tf.shape(lstm_logits)
        #self.lstm_h_shape = self.lstm.hidden_shape
        #self.lstm_h_r_shape = self.lstm.hidden_reshape_shape
        #self.lstm_output_shape = self.lstm.outputs_shape
        #self.lstm_output_shape_T = self.lstm.outputs_T_shape
        #self.lstm_outputs_fw = self.lstm.outputs_fw
        #self.lstm_outputs_bw = self.lstm.outputs_bw

        self.lstm_hidden = self.lstm.hidden
        self.last_lstm_h = self.lstm.last_h
        self.lstm_hidden = tf.reshape(self.lstm_hidden, [-1, params.hidden_size])
        
        #self.h_comb = self.last_lstm_h
        #self.h_comb = self.h

        ## TODO: Weighted sum or weighted projection into a common space
        if params.common_space:
            # Apply activation on DocNADE hidden vectors
            if params.activation == 'sigmoid':
                h = tf.sigmoid(h)
                self.h = tf.sigmoid(self.h)
            elif params.activation == 'tanh':
                h = tf.tanh(h)
                self.h = tf.tanh(self.h)
            elif params.activation == 'relu':
                h = tf.nn.relu(h)
                self.h = tf.nn.relu(self.h)
            else:
                print('Invalid value for activation: %s' % (params.activation))
                exit()

            h_docnade_reshaped = tf.reshape(h, [-1, params.hidden_size])
            h_lstm_reshaped = tf.reshape(self.lstm_hidden, [-1, params.hidden_size])
            
            h_docnade_temp = tf.matmul(h_docnade_reshaped, W_proj_docnade)
            h_lstm_temp = tf.matmul(h_lstm_reshaped, W_proj_lstm)

            if params.activation == 'sigmoid':
                h_comb = tf.sigmoid(tf.nn.bias_add(tf.add(h_docnade_temp, h_lstm_temp), bias_proj), name='hidden_comb_common_space')
                h_lstm = tf.sigmoid(tf.nn.bias_add(h_lstm_temp, bias_proj), name='hidden_lstm_common_space')
                h_docnade = tf.sigmoid(tf.nn.bias_add(h_docnade_temp, bias_proj), name='hidden_docnade_common_space')
                
                self.h_comb_last = tf.sigmoid(tf.nn.bias_add(tf.add(tf.matmul(self.h, W_proj_docnade), tf.matmul(self.last_lstm_h, W_proj_lstm)), bias_proj), name='last_hidden_comb_common_space')
                self.last_lstm_h = tf.sigmoid(tf.nn.xw_plus_b(self.last_lstm_h, W_proj_lstm, bias_proj), name='last_hidden_lstm_common_space')
                self.h = tf.sigmoid(tf.nn.xw_plus_b(self.h, W_proj_docnade, bias_proj), name='last_hidden_docnade_common_space')
            elif params.activation == 'tanh':
                h_comb = tf.tanh(tf.nn.bias_add(tf.add(h_docnade_temp, h_lstm_temp), bias_proj), name='hidden_comb_common_space')
                h_lstm = tf.tanh(tf.nn.bias_add(h_lstm_temp, bias_proj), name='hidden_lstm_common_space')
                h_docnade = tf.tanh(tf.nn.bias_add(h_docnade_temp, bias_proj), name='hidden_docnade_common_space')

                self.h_comb_last = tf.tanh(tf.nn.bias_add(tf.add(tf.matmul(self.h, W_proj_docnade), tf.matmul(self.last_lstm_h, W_proj_lstm)), bias_proj), name='last_hidden_comb_common_space')
                self.last_lstm_h = tf.tanh(tf.nn.xw_plus_b(self.last_lstm_h, W_proj_lstm, bias_proj), name='last_hidden_lstm_common_space')
                self.h = tf.tanh(tf.nn.xw_plus_b(self.h, W_proj_docnade, bias_proj), name='last_hidden_docnade_common_space')
            elif params.activation == 'relu':
                h_comb = tf.nn.relu(tf.nn.bias_add(tf.add(h_docnade_temp, h_lstm_temp), bias_proj), name='hidden_comb_common_space')
                h_lstm = tf.nn.relu(tf.nn.bias_add(h_lstm_temp, bias_proj), name='hidden_lstm_common_space')
                h_docnade = tf.nn.relu(tf.nn.bias_add(h_docnade_temp, bias_proj), name='hidden_docnade_common_space')

                self.h_comb_last = tf.nn.relu(tf.nn.bias_add(tf.add(tf.matmul(self.h, W_proj_docnade), tf.matmul(self.last_lstm_h, W_proj_lstm)), bias_proj), name='last_hidden_comb_common_space')
                self.last_lstm_h = tf.nn.relu(tf.nn.xw_plus_b(self.last_lstm_h, W_proj_lstm, bias_proj), name='last_hidden_lstm_common_space')
                self.h = tf.nn.relu(tf.nn.xw_plus_b(self.h, W_proj_docnade, bias_proj), name='last_hidden_docnade_common_space')
            else:
                print('Invalid value for activation: %s' % (params.activation))
                exit()

            #h_comb_temp = tf.reshape(h_comb, [self.b_s_x[0], self.b_s_x[1], params.hidden_size])
            #h_docnade_temp = tf.reshape(h_docnade, [self.b_s_x[0], self.b_s_x[1], params.hidden_size])
            #h_lstm_temp = tf.reshape(h_lstm, [self.b_s_lstm[0], self.b_s_lstm[1], params.hidden_size])

            #self.h_comb_last = tf.gather_nd(h_comb_temp, indices)
            #self.h = tf.gather_nd(h_docnade_temp, indices)
            #self.h_lstm_last = tf.gather_nd(h_lstm_temp, lstm_indices) ## lstm_indices OR indices??

        else:
            # Apply activation
            if params.activation == 'sigmoid':
                h = tf.sigmoid(h + bias)
                self.h = tf.sigmoid(self.h + bias)
            elif params.activation == 'tanh':
                h = tf.tanh(h + bias)
                self.h = tf.tanh(self.h + bias)
            elif params.activation == 'relu':
                h = tf.nn.relu(h + bias)
                self.h = tf.nn.relu(self.h + bias)
            else:
                print('Invalid value for activation: %s' % (params.activation))
                exit()

            """
            h_comb = tf.add(h, self.lstm_hidden)
            h_comb_last = tf.add(self.h, self.last_lstm_h)

            # Apply activation
            if params.activation == 'sigmoid':
                h_comb = tf.sigmoid(h_comb + bias)
                h_lstm = tf.sigmoid(self.lstm_hidden)
                h_docnade = tf.sigmoid(h + bias)

                self.h_comb_last = tf.sigmoid(h_comb_last + bias, name='last_hidden_comb')
                self.last_lstm_h = tf.sigmoid(self.last_lstm_h, name='last_hidden_lstm')
                self.h = tf.sigmoid(self.h + bias, name='last_hidden_docnade')
            elif params.activation == 'tanh':
                h_comb = tf.tanh(h_comb + bias)
                h_lstm = tf.tanh(self.lstm_hidden)
                h_docnade = tf.tanh(h + bias)

                self.h_comb_last = tf.tanh(h_comb_last + bias, name='last_hidden_comb')
                self.last_lstm_h = tf.tanh(self.last_lstm_h, name='last_hidden_lstm')
                self.h = tf.tanh(self.h + bias, name='last_hidden_docnade')
            elif params.activation == 'relu':
                h_comb = tf.nn.relu(h_comb + bias)
                h_lstm = tf.nn.relu(self.lstm_hidden)
                h_docnade = tf.nn.relu(h + bias)

                self.h_comb_last = tf.nn.relu(h_comb_last + bias, name='last_hidden_comb')
                self.last_lstm_h = tf.nn.relu(self.last_lstm_h, name='last_hidden_lstm')
                self.h = tf.nn.relu(self.h + bias, name='last_hidden_docnade')
            else:
                print('Invalid value for activation: %s' % (params.activation))
                exit()

            #h_comb = tf.reshape(h_comb, [-1, params.hidden_size], name='hidden_comb')
            #h_lstm = tf.reshape(h_lstm, [-1, params.hidden_size], name='hidden_lstm')
            #h_docnade = tf.reshape(h_docnade, [-1, params.hidden_size], name='hidden_docnade')
            """

        lstm_hidden_reduced = tf.scalar_mul(self.lambda_hidden_lstm, self.lstm_hidden)
        last_lstm_h_reduced = tf.scalar_mul(self.lambda_hidden_lstm, self.last_lstm_h)
        
        h_comb = tf.add(h, lstm_hidden_reduced, name='combined_hidden_docnade_lstm')
        h_docnade = h
        h_lstm = self.lstm_hidden

        self.h_comb_concat = tf.concat([
            self.h, self.last_lstm_h
        ], axis=1, name='h_comb_concat')
        self.h_comb_sum = tf.add(self.h, last_lstm_h_reduced, name='h_comb_sum')

        self.h_reshaped_shape = tf.shape(h_comb)

        #########################################################################
        if params.supervised:
            if params.combination_type == "concat":
                self.disc_h = self.h_comb_concat
            elif params.combination_type == "sum":
                self.disc_h = self.h_comb_sum
            elif params.combination_type == "projected":
                self.disc_h = self.h_comb_last
            
            
            max_U_init = 1.0 / (params.hidden_size * params.num_classes)

            U = tf.get_variable(
                'U_disc',
                [params.hidden_size, params.num_classes],
                initializer=tf.random_uniform_initializer(
                    maxval=max_U_init
                )
            )

            d = tf.get_variable(
                'd_disc',
                [params.num_classes],
                initializer=tf.constant_initializer(0)
            )

            disc_logits = tf.nn.xw_plus_b(self.disc_h, U, d, name='disc_logits')
            one_hot_labels = tf.one_hot(y, depth=params.num_classes)

            self.pred_labels = tf.argmax(disc_logits, axis=1)

            self.disc_loss = tf.Variable(0.0, name='disc_loss')
            self.disc_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=one_hot_labels,
                logits=disc_logits,
            )

            self.disc_accuracy = tf.metrics.accuracy(self.y, self.pred_labels, name='disc_accuracy')

            self.disc_output = disc_logits
            """
            self.mlp = MLP(input=self.disc_h, labels=y, num_classes=params.num_classes, hidden_sizes=params.hidden_sizes)
            self.disc_output = self.mlp.disc_output
            self.disc_loss = self.mlp.disc_loss
            self.disc_accuracy = self.mlp.disc_accuracy
            """
        #####################################################################

        if not params.num_samples:
            self.comb_logits, self.docnade_logits, self.lstm_logits = linear(h_comb, h_docnade, h_lstm, params.vocab_size, 'softmax', W_initializer=None)
            loss_function = None
        else:
            self.comb_logits, self.docnade_logits, self.lstm_logits = linear(h_comb, h_docnade, h_lstm, params.num_samples, 'softmax', W_initializer=None)

            if W_initializer_docnade is None:
                max_embed_init = 1.0 / (params.vocab_size * params.hidden_size)
                w_t = tf.get_variable(
                    "proj_w_t",
                    [params.vocab_size, params.num_samples],
                    initializer=tf.random_uniform_initializer(
                        maxval=max_embed_init)
                )
            else:
                w_t = tf.get_variable(
                    "proj_w_t",
                    # [params.vocab_size, params.num_samples],
                    initializer=W_initializer_docnade
                )
            b = tf.get_variable("proj_b", [params.vocab_size])
            self.proj_w = tf.transpose(w_t)
            self.proj_b = b

            def sampled_loss(logits, labels):
                labels = tf.reshape(labels, [-1, 1])
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(logits, tf.float32)
                return tf.nn.sampled_softmax_loss(
                    weights=local_w_t,
                    biases=local_b,
                    labels=labels,
                    inputs=local_inputs,
                    num_sampled=params.num_samples,
                    num_classes=params.vocab_size,
                    partition_strategy='div'
                )
            loss_function = sampled_loss

        #self.logits = tf.add(self.docnade_logits, self.lstm_logits)
        #self.logits_shape = tf.shape(self.logits)

        # Compute the loss. If using sampled softmax for training, use full
        # softmax for evaluation and validation
        #if not params.num_samples:
        #    self.loss = masked_sequence_cross_entropy_loss(
        #        x,
        #        seq_lengths,
        #        self.logits
        #    )
        #else:
        #    projected_logits = \
        #        tf.matmul(self.logits, self.proj_w) + self.proj_b
        #    self.loss = masked_sequence_cross_entropy_loss(
        #        x,
        #        seq_lengths,
        #        projected_logits
        #    )

        self.loss_normed, self.labels, self.mask, self.loss_unnormed  = masked_sequence_cross_entropy_loss(
            x,
            seq_lengths,
            self.comb_logits,
            loss_function=loss_function,
            norm_by_seq_lengths=True,
            name='docnade_loss_comb'
        )

        self.loss_normed_docnade, _, _, self.loss_unnormed_docnade  = masked_sequence_cross_entropy_loss(
            x,
            seq_lengths,
            self.docnade_logits,
            loss_function=loss_function,
            norm_by_seq_lengths=True,
            name='docnade_loss_docnade'
        )

        self.loss_normed_lstm, _, _, self.loss_unnormed_lstm  = masked_sequence_cross_entropy_loss(
            x,
            seq_lengths,
            self.lstm_logits,
            loss_function=loss_function,
            norm_by_seq_lengths=True,
            name='docnade_loss_lstm'
        )

        #self.total_loss = tf.Variable(0.0, trainable=False, name='total_loss')
        self.total_loss = self.loss_unnormed

        if params.include_lstm_loss:
            #self.total_loss = self.docnade_loss_weight * self.total_loss + self.lstm_loss_weight * self.lstm.loss_normed
            self.total_loss = self.docnade_loss_weight * self.total_loss + self.lstm_loss_weight * self.lstm.loss_unnormed

        if params.supervised:
            self.total_loss = params.generative_loss_weight * self.total_loss + self.disc_loss

        # Optimiser
        #step = tf.Variable(0, trainable=False)
        self.opt = tf.train.AdamOptimizer(learning_rate=params.learning_rate).minimize(self.total_loss)
        #self.opt = tf.train.GradientDescentOptimizer(learning_rate=params.learning_rate).minimize(self.total_loss)
        """
        optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
        self.opt = gradients(
            opt=optimizer,
            loss=self.total_loss,
            vars=tf.trainable_variables(),
            step=step
        )
        """
        tf.add_to_collection("optimizer", self.opt)


class BiRnnLstm_reload(object):
    def __init__(self, input, rnn_seq_lengths, transition_matrix, hidden_size, vocab_size, batch_size, indices,
                 rnn_vocab_length, trainable_val, max_seq_len, include_lstm_loss, embeddings_rnn_reload=None,
                 V_rnn_reload=None, d_rnn_reload=None, W_initializer_lstm=None):
        self.input = input
        self.transition_matrix = transition_matrix

        # Do an embedding lookup for each word in each sequence
        # Initialisation scheme taken from the original DocNADE source
        W = tf.get_variable(
            'embeddings_rnn',
            initializer=embeddings_rnn_reload,
            trainable=False
        )

        self.embeddings = tf.nn.embedding_lookup(W, input)
        embedding_input = self.embeddings
        """
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, timesteps, n_input)
        # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)
        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
        #max_seq_len = 59
        #new_input = tf.unstack(input, max_seq_len, 1)

        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = rnn.BasicLSTMCell(hidden_size, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = rnn.BasicLSTMCell(hidden_size, forget_bias=1.0)

        # Get lstm cell output
        #try:
        #    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, new_input,
        #                                        dtype=tf.float32)
        #except Exception: # Old TensorFlow version only returns outputs not states
        #    outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, new_input,
        #                                        dtype=tf.float32)

        #try:
        #    outputs, _, _ = rnn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, new_input,
        #                                        dtype=tf.float32)
        #except Exception: # Old TensorFlow version only returns outputs not states
        outputs_rnn, final_states_rnn = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, input,
                                            dtype=tf.float32)

        outputs_fw, outputs_bw = outputs_rnn
        final_states_fw, final_states_bw = final_states_rnn

        #outputs = tf.concat([
        #    outputs_fw, outputs_bw
        #], axis=2)

        self.outputs_fw = outputs_fw
        self.outputs_bw = outputs_bw

        #self.outputs = outputs
        #self.outputs_shape = tf.shape(outputs)
        #outputs_T = tf.transpose(outputs, [0,1,2])
        #self.outputs_T_shape = tf.shape(outputs_T)

        #hidden = tf.matmul(self.transition_matrix, outputs_T)
        hidden_fw = tf.matmul(self.transition_matrix, outputs_fw)
        hidden_bw = tf.matmul(self.transition_matrix, outputs_bw)
        hidden = tf.concat([
            hidden_fw, hidden_bw
        ], axis=2)
        self.hidden_shape = tf.shape(hidden)

        self.last_h_fw = tf.gather_nd(hidden_fw, indices)
        self.last_h_bw = hidden_bw[:, 0, :]

        #self.last_h = tf.concat([
        #    self.last_h_fw, self.last_h_bw
        #], axis=1)
        self.last_h = self.last_h_fw

        hidden_reshape = tf.reshape(hidden, [-1, 2*hidden_size])
        self.hidden_reshape_shape = tf.shape(hidden_reshape)

        with tf.device('/cpu:0'):
            max_embed_init = 1.0 / (2 * hidden_size * vocab_size)
            self.W = tf.get_variable(
                'out',
                [2*hidden_size, vocab_size],
                dtype=tf.float32,
                initializer=tf.random_uniform_initializer(
                    maxval=max_embed_init
                )
            )

        self.logits = tf.matmul(hidden_reshape, self.W)
        """
        #max_seq_len = 88
        #max_seq_len = 14812

        #def linear_activation(x):
        #    return x

        new_input = tf.unstack(embedding_input, max_seq_len, 1)
        #lstm_cell = rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, activation=linear_activation)
        lstm_cell = rnn.BasicLSTMCell(hidden_size, forget_bias=1.0)
        outputs, state = rnn.static_rnn(lstm_cell, new_input, dtype=tf.float32)

        self.outputs = outputs
        self.outputs_shape = tf.shape(outputs)
        outputs_T = tf.transpose(outputs, [1,0,2])
        self.outputs_T_shape = tf.shape(outputs_T)

        self.last_h = tf.gather_nd(outputs_T, indices)

        outputs_T_new = tf.concat([
            outputs_T, tf.zeros([batch_size, 1, hidden_size])
        ], axis=1)
        self.outputs_T = outputs_T

        #hidden = tf.matmul(self.transition_matrix, outputs_T)
        elems = (outputs_T_new, self.transition_matrix)
        hidden = tf.map_fn(lambda inputs: tf.nn.embedding_lookup(inputs[0], inputs[1]), elems, dtype=tf.float32)
        #hidden = tf.nn.embedding_lookup(outputs_T, self.transition_matrix)[0]
        hidden = tf.reduce_sum(hidden, axis=2)

        #outputs_T = outputs_T[:, :-1, :]

        hidden = tf.concat([
            tf.zeros([batch_size, 1, hidden_size], dtype=tf.float32), hidden
        ], axis=1)
        hidden = hidden[:, :-1, :]
        self.hidden = hidden

        if include_lstm_loss:
            #self.hidden_reshape = tf.reshape(hidden, [-1, hidden_size])
            self.outputs_T_reshape = tf.reshape(outputs_T, [-1, hidden_size])

            
            max_embed_init = 1.0 / (hidden_size * rnn_vocab_length)
            self.V_rnn = tf.get_variable(
                'V_rnn',
                dtype=tf.float32,
                initializer=V_rnn_reload,
                trainable=False
            )

            self.bias_rnn = tf.get_variable(
                'd_rnn',
                initializer=d_rnn_reload,
                trainable=False
            )

            #self.logits = tf.matmul(self.hidden_reshape, self.W)
            self.logits = tf.nn.xw_plus_b(self.outputs_T_reshape, self.V_rnn, self.bias_rnn)

            self.loss_normed, _, _, self.loss_unnormed  = masked_sequence_cross_entropy_loss(
                input,
                rnn_seq_lengths,
                self.logits,
                loss_function=None,
                norm_by_seq_lengths=True
            )


class DocNADE_reload(object):
    def __init__(self, x, y, seq_lengths, params, x_rnn, rnn_seq_lengths, rnn_transition_matrix, rnn_vocab_length,
                 max_seq_len, proj_docnade_reload=None, proj_lstm_reload=None, bias_proj_reload=None, embeddings_docnade_reload=None,
                 bias_docnade_reload=None, V_reload=None, b_reload=None, embeddings_rnn_reload=None, V_rnn_reload=None, d_rnn_reload=None,
                 W_initializer_docnade=None):
        print("\n\nDocNADE with RNN-LSTM\n\n")
        self.x = x
        self.y = y
        self.seq_lengths = seq_lengths
        self.max_seq_len = max_seq_len

        self.x_rnn = x_rnn
        self.rnn_seq_lengths = rnn_seq_lengths
        self.rnn_transition_matrix = rnn_transition_matrix

        batch_size = tf.shape(x)[0]

        self.b_s_x = tf.shape(x)
        self.b_s_lstm = tf.shape(x_rnn)

        if params.common_space:
            # Initialisation scheme taken from the original DocNADE source
            max_embed_init = 1.0 / (params.hidden_size * params.hidden_size)
            W_proj_docnade = tf.get_variable(
                'proj_docnade',
                initializer=proj_docnade_reload,
                trainable=False
            )

            W_proj_lstm = tf.get_variable(
                'proj_lstm',
                initializer=proj_lstm_reload,
                trainable=False
            )

            bias_proj = tf.get_variable(
                'bias_proj',
                initializer=bias_proj_reload,
                trainable=False
            )

        # Do an embedding lookup for each word in each sequence
        # Initialisation scheme taken from the original DocNADE source
        W = tf.get_variable(
            'embeddings_docnade',
            initializer=embeddings_docnade_reload,
            trainable=False
        )
        self.embeddings = tf.nn.embedding_lookup(W, x)

        self.embeddings_shape = tf.shape(self.embeddings)

        # Compute the hidden layer inputs: each gets summed embeddings of
        # previous words
        def sum_embeddings(previous, current):
            return previous + current

        h = tf.scan(sum_embeddings, tf.transpose(self.embeddings, [1, 2, 0]))
        self.h_shape = tf.shape(h)
        h = tf.transpose(h, [2, 0, 1])
        self.h_transpose_shape = tf.shape(h)

        bias = tf.get_variable(
            'bias_docnade',
            initializer=bias_docnade_reload,
            trainable=False
        )

        #bias_lstm = tf.get_variable(
        #    'bias_lstm',
        #    [params.hidden_size],
        #    initializer=tf.constant_initializer(0)
        #)

        self.bias_shape = tf.shape(bias)

        # add initial zero vector to each sequence, will then generate the
        # first element using just the bias term
        h = tf.concat([
            tf.zeros([batch_size, 1, params.hidden_size], dtype=tf.float32), h
        ], axis=1)
        
        self.pre_act = h
        self.pre_act_shape = tf.shape(self.pre_act)

        ## Apply activation
        #if params.activation == 'sigmoid':
        #    h_docnade = tf.sigmoid(h + bias)
        #elif params.activation == 'tanh':
        #    h_docnade = tf.tanh(h + bias)
        #elif params.activation == 'relu':
        #    h_docnade = tf.nn.relu(h + bias)
        #else:
        #    print('Invalid value for activation: %s' % (params.activation))
        #    exit()
        #self.aft_act = h_docnade

        # Extract final state for each sequence in the batch
        indices = tf.stack([
            tf.range(batch_size),
            tf.to_int32(seq_lengths)
        ], axis=1)
        self.indices = indices
        self.h = tf.gather_nd(h, indices)

        #h_docnade = h_docnade[:, :-1, :]
        h = h[:, :-1, :]

        self.final_state_h_shape = tf.shape(self.h)

        #########################################################################
        ######################## ADDING LSTM HIDDEN LAYER #######################

        lstm_indices = tf.stack([
            tf.range(batch_size),
            tf.to_int32(rnn_seq_lengths) - 1
        ], axis=1)
        self.lstm_indices = lstm_indices

        #input_rnn = tf.nn.embedding_lookup(glove_embeddings, x_rnn)

        self.lstm = BiRnnLstm_reload(x_rnn, rnn_seq_lengths, rnn_transition_matrix, params.hidden_size, params.vocab_size, batch_size, lstm_indices,
                              rnn_vocab_length, params.update_rnn_w, max_seq_len, params.include_lstm_loss, embeddings_rnn_reload=embeddings_rnn_reload,
                              V_rnn_reload=V_rnn_reload, d_rnn_reload=d_rnn_reload, W_initializer_lstm=None)
        #self.lstm_logits = self.lstm.logits
        #self.lstm_logits_shape = tf.shape(lstm_logits)
        #self.lstm_h_shape = self.lstm.hidden_shape
        #self.lstm_h_r_shape = self.lstm.hidden_reshape_shape
        #self.lstm_output_shape = self.lstm.outputs_shape
        #self.lstm_output_shape_T = self.lstm.outputs_T_shape
        #self.lstm_outputs_fw = self.lstm.outputs_fw
        #self.lstm_outputs_bw = self.lstm.outputs_bw

        self.lstm_hidden = self.lstm.hidden
        self.last_lstm_h = self.lstm.last_h
        
        #self.h_comb = self.last_lstm_h
        #self.h_comb = self.h

        ## TODO: Weighted sum or weighted projection into a common space
        if params.common_space:
            # Apply activation on DocNADE hidden vectors
            if params.activation == 'sigmoid':
                h = tf.sigmoid(h)
                self.h = tf.sigmoid(self.h)
            elif params.activation == 'tanh':
                h = tf.tanh(h)
                self.h = tf.tanh(self.h)
            elif params.activation == 'relu':
                h = tf.nn.relu(h)
                self.h = tf.nn.relu(self.h)
            else:
                print('Invalid value for activation: %s' % (params.activation))
                exit()

            h_docnade_reshaped = tf.reshape(h, [-1, params.hidden_size])
            h_lstm_reshaped = tf.reshape(self.lstm_hidden, [-1, params.hidden_size])
            
            h_docnade_temp = tf.matmul(h_docnade_reshaped, W_proj_docnade)
            h_lstm_temp = tf.matmul(h_lstm_reshaped, W_proj_lstm)

            if params.activation == 'sigmoid':
                h_comb = tf.sigmoid(tf.nn.bias_add(tf.add(h_docnade_temp, h_lstm_temp), bias_proj))
                h_lstm = tf.sigmoid(tf.nn.bias_add(h_lstm_temp, bias_proj))
                h_docnade = tf.sigmoid(tf.nn.bias_add(h_docnade_temp, bias_proj))
                
                self.h_comb_last = tf.sigmoid(tf.nn.bias_add(tf.add(tf.matmul(self.h, W_proj_docnade), tf.matmul(self.last_lstm_h, W_proj_lstm)), bias_proj))
                self.last_lstm_h = tf.sigmoid(tf.nn.xw_plus_b(self.last_lstm_h, W_proj_lstm, bias_proj))
                self.h = tf.sigmoid(tf.nn.xw_plus_b(self.h, W_proj_docnade, bias_proj))
            elif params.activation == 'tanh':
                h_comb = tf.tanh(tf.nn.bias_add(tf.add(h_docnade_temp, h_lstm_temp), bias_proj))
                h_lstm = tf.tanh(tf.nn.bias_add(h_lstm_temp, bias_proj))
                h_docnade = tf.tanh(tf.nn.bias_add(h_docnade_temp, bias_proj))

                self.h_comb_last = tf.tanh(tf.nn.bias_add(tf.add(tf.matmul(self.h, W_proj_docnade), tf.matmul(self.last_lstm_h, W_proj_lstm)), bias_proj))
                self.last_lstm_h = tf.tanh(tf.nn.xw_plus_b(self.last_lstm_h, W_proj_lstm, bias_proj))
                self.h = tf.tanh(tf.nn.xw_plus_b(self.h, W_proj_docnade, bias_proj))
            elif params.activation == 'relu':
                h_comb = tf.nn.relu(tf.nn.bias_add(tf.add(h_docnade_temp, h_lstm_temp), bias_proj))
                h_lstm = tf.nn.relu(tf.nn.bias_add(h_lstm_temp, bias_proj))
                h_docnade = tf.nn.relu(tf.nn.bias_add(h_docnade_temp, bias_proj))

                self.h_comb_last = tf.nn.relu(tf.nn.bias_add(tf.add(tf.matmul(self.h, W_proj_docnade), tf.matmul(self.last_lstm_h, W_proj_lstm)), bias_proj))
                self.last_lstm_h = tf.nn.relu(tf.nn.xw_plus_b(self.last_lstm_h, W_proj_lstm, bias_proj))
                self.h = tf.nn.relu(tf.nn.xw_plus_b(self.h, W_proj_docnade, bias_proj))
            else:
                print('Invalid value for activation: %s' % (params.activation))
                exit()

            #h_comb_temp = tf.reshape(h_comb, [self.b_s_x[0], self.b_s_x[1], params.hidden_size])
            #h_docnade_temp = tf.reshape(h_docnade, [self.b_s_x[0], self.b_s_x[1], params.hidden_size])
            #h_lstm_temp = tf.reshape(h_lstm, [self.b_s_lstm[0], self.b_s_lstm[1], params.hidden_size])

            #self.h_comb_last = tf.gather_nd(h_comb_temp, indices)
            #self.h = tf.gather_nd(h_docnade_temp, indices)
            #self.h_lstm_last = tf.gather_nd(h_lstm_temp, lstm_indices) ## lstm_indices OR indices??

        else:
            h_comb = tf.add(h, self.lstm_hidden)
            h_comb_last = tf.add(self.h, self.last_lstm_h)

            # Apply activation
            if params.activation == 'sigmoid':
                h_comb = tf.sigmoid(h_comb + bias)
                h_lstm = tf.sigmoid(self.lstm_hidden)
                h_docnade = tf.sigmoid(h + bias)

                self.h_comb_last = tf.sigmoid(h_comb_last + bias)
                self.last_lstm_h = tf.sigmoid(self.last_lstm_h)
                self.h = tf.sigmoid(self.h + bias)
            elif params.activation == 'tanh':
                h_comb = tf.tanh(h_comb + bias)
                h_lstm = tf.tanh(self.lstm_hidden)
                h_docnade = tf.tanh(h + bias)

                self.h_comb_last = tf.tanh(h_comb_last + bias)
                self.last_lstm_h = tf.tanh(self.last_lstm_h)
                self.h = tf.tanh(self.h + bias)
            elif params.activation == 'relu':
                h_comb = tf.nn.relu(h_comb + bias)
                h_lstm = tf.nn.relu(self.lstm_hidden)
                h_docnade = tf.nn.relu(h + bias)

                self.h_comb_last = tf.nn.relu(h_comb_last + bias)
                self.last_lstm_h = tf.nn.relu(self.last_lstm_h)
                self.h = tf.nn.relu(self.h + bias)
            else:
                print('Invalid value for activation: %s' % (params.activation))
                exit()

            h_comb = tf.reshape(h_comb, [-1, params.hidden_size])
            h_lstm = tf.reshape(h_lstm, [-1, params.hidden_size])
            h_docnade = tf.reshape(h_docnade, [-1, params.hidden_size])

        self.h_comb_concat = tf.concat([
            self.h, self.last_lstm_h
        ], axis=1)
        self.h_comb_sum = tf.add(self.h, self.last_lstm_h)

        self.h_reshaped_shape = tf.shape(h_comb)

        #########################################################################
        if params.supervised:
            if params.combination_type == "concat":
                self.disc_h = self.h_comb_concat
            elif params.combination_type == "sum":
                self.disc_h = self.h_comb_sum
            elif params.combination_type == "projected":
                self.disc_h = self.h_comb_last
            
            
            max_U_init = 1.0 / (params.hidden_size * params.num_classes)

            U = tf.get_variable(
                'U_disc',
                [params.hidden_size, params.num_classes],
                initializer=tf.random_uniform_initializer(
                    maxval=max_U_init
                )
            )

            d = tf.get_variable(
                'd_disc',
                [params.num_classes],
                initializer=tf.constant_initializer(0)
            )

            disc_logits = tf.nn.xw_plus_b(self.disc_h, U, d)
            one_hot_labels = tf.one_hot(y, depth=params.num_classes)

            self.pred_labels = tf.argmax(disc_logits, axis=1)

            self.disc_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=one_hot_labels,
                logits=disc_logits,
            )

            self.disc_accuracy = tf.metrics.accuracy(self.y, self.pred_labels)

            self.disc_output = disc_logits
            """
            self.mlp = MLP(input=self.disc_h, labels=y, num_classes=params.num_classes, hidden_sizes=params.hidden_sizes)
            self.disc_output = self.mlp.disc_output
            self.disc_loss = self.mlp.disc_loss
            self.disc_accuracy = self.mlp.disc_accuracy
            """
        #####################################################################

        if not params.num_samples:
            self.comb_logits, self.docnade_logits, self.lstm_logits = linear_reload(h_comb, h_docnade, h_lstm, params.vocab_size, 'softmax',  V_reload=V_reload, b_reload=b_reload)
            loss_function = None
        else:
            self.comb_logits, self.docnade_logits, self.lstm_logits = linear_reload(h_comb, h_docnade, h_lstm, params.num_samples, 'softmax', V_reload=V_reload, b_reload=b_reload)

            if W_initializer_docnade is None:
                max_embed_init = 1.0 / (params.vocab_size * params.hidden_size)
                w_t = tf.get_variable(
                    "proj_w_t",
                    [params.vocab_size, params.num_samples],
                    initializer=tf.random_uniform_initializer(
                        maxval=max_embed_init)
                )
            else:
                w_t = tf.get_variable(
                    "proj_w_t",
                    # [params.vocab_size, params.num_samples],
                    initializer=W_initializer_docnade
                )
            b = tf.get_variable("proj_b", [params.vocab_size])
            self.proj_w = tf.transpose(w_t)
            self.proj_b = b

            def sampled_loss(logits, labels):
                labels = tf.reshape(labels, [-1, 1])
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(logits, tf.float32)
                return tf.nn.sampled_softmax_loss(
                    weights=local_w_t,
                    biases=local_b,
                    labels=labels,
                    inputs=local_inputs,
                    num_sampled=params.num_samples,
                    num_classes=params.vocab_size,
                    partition_strategy='div'
                )
            loss_function = sampled_loss

        #self.logits = tf.add(self.docnade_logits, self.lstm_logits)
        #self.logits_shape = tf.shape(self.logits)

        # Compute the loss. If using sampled softmax for training, use full
        # softmax for evaluation and validation
        #if not params.num_samples:
        #    self.loss = masked_sequence_cross_entropy_loss(
        #        x,
        #        seq_lengths,
        #        self.logits
        #    )
        #else:
        #    projected_logits = \
        #        tf.matmul(self.logits, self.proj_w) + self.proj_b
        #    self.loss = masked_sequence_cross_entropy_loss(
        #        x,
        #        seq_lengths,
        #        projected_logits
        #    )

        self.loss_normed, self.labels, self.mask, self.loss_unnormed  = masked_sequence_cross_entropy_loss(
            x,
            seq_lengths,
            self.comb_logits,
            loss_function=loss_function,
            norm_by_seq_lengths=True
        )

        self.loss_normed_docnade, _, _, self.loss_unnormed_docnade  = masked_sequence_cross_entropy_loss(
            x,
            seq_lengths,
            self.docnade_logits,
            loss_function=loss_function,
            norm_by_seq_lengths=True
        )

        self.loss_normed_lstm, _, _, self.loss_unnormed_lstm  = masked_sequence_cross_entropy_loss(
            x,
            seq_lengths,
            self.lstm_logits,
            loss_function=loss_function,
            norm_by_seq_lengths=True
        )

        self.total_loss = self.loss_normed

        if params.include_lstm_loss:
            self.total_loss = self.total_loss + self.lstm.loss_normed

        if params.supervised:
            self.total_loss = params.generative_loss_weight * self.total_loss + self.disc_loss
