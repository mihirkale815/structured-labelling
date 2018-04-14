
import tensorflow as tf
from neural import dynamicBiRNN, LReLu, MLP, get_structure
import numpy as np



class StructureModelCRF():
    def __init__(self, config):
        self.config = config
        t_variables = {}
        t_variables['keep_prob'] = tf.placeholder(tf.float32)
        t_variables['batch_l'] = tf.placeholder(tf.int32)
        t_variables['token_idxs'] = tf.placeholder(tf.int32, [None, None])
        t_variables['verb_indicator'] = tf.placeholder(tf.float32, [None, None])
        t_variables['sent_l'] = tf.placeholder(tf.int32, [None])
        t_variables['doc_l'] = tf.placeholder(tf.int32, [None])
        t_variables['max_sent_l'] = tf.placeholder(tf.int32)
        t_variables['max_doc_l'] = tf.placeholder(tf.int32)
        t_variables['gold_labels'] = tf.placeholder(tf.int32, [None,None])
        t_variables['mask_tokens'] = tf.placeholder(tf.float32, [None, None])
        t_variables['mask_sents'] = tf.placeholder(tf.float32, [None, None])
        #t_variables['mask_parser_1'] = tf.placeholder(tf.float32, [None, None, None])
        #t_variables['mask_parser_2'] = tf.placeholder(tf.float32, [None, None, None])
        self.t_variables = t_variables


    def get_feed_dict(self, batch,id2tag=None):
        batch_size = len(batch)
        #doc_l_matrix = np.zeros([batch_size], np.int32)
        #for i, instance in enumerate(batch):
        #    n_sents = len(instance.token_idxs)
        #    doc_l_matrix[i] = n_sents
        #max_doc_l = np.max(doc_l_matrix)
        max_sent_l = max([len(sen.token_idxs) for sen in batch])
        token_idxs_matrix = np.zeros([batch_size, max_sent_l], np.int32)
        verb_indicator_matrix = np.zeros([batch_size, max_sent_l], np.int32)
        sent_l_matrix = np.zeros([batch_size], np.int32)
        gold_matrix = np.zeros([batch_size, max_sent_l], np.int32)
        mask_tokens_matrix = np.ones([batch_size,max_sent_l], np.float32)
        #mask_sents_matrix = np.ones([batch_size, max_doc_l], np.float32)
        for i, instance in enumerate(batch):
            sequence_padded, sequence_length = pad_sequence(instance.goldLabel,0,max_sent_l)

            gold_matrix[i] = np.array(sequence_padded)


            verb_indicator = np.array([1 if id2tag[k] in ('b-v') else 0 for k in sequence_padded])
            #print("verb indicator sum = ", verb_indicator.sum())
            verb_indicator_matrix[i, :] = verb_indicator



            sequence_padded, sequence_length = pad_sequence(instance.token_idxs, 0, max_sent_l)

            token_idxs_matrix[i,:] = np.asarray(sequence_padded)
            mask_tokens_matrix[i, len(instance.token_idxs):] = 0
            sent_l_matrix[i] = len(instance.token_idxs)




            #mask_sents_matrix[i, n_sents:] = 0
        #mask_parser_1 = np.ones([batch_size, max_doc_l, max_doc_l], np.float32)
        #mask_parser_2 = np.ones([batch_size, max_doc_l, max_doc_l], np.float32)
        #mask_parser_1[:, :, 0] = 0
        #mask_parser_2[:, 0, :] = 0
        if (self.config.large_data):
            if (batch_size  * max_sent_l * max_sent_l > 16 * 200000):
                return [batch_size  * max_sent_l * max_sent_l / (16 * 200000) + 1]

        feed_dict = {self.t_variables['token_idxs']: token_idxs_matrix, self.t_variables['sent_l']: sent_l_matrix,
                     self.t_variables['mask_tokens']: mask_tokens_matrix,#self.t_variables['mask_sents']: mask_sents_matrix,
                     self.t_variables['gold_labels']: gold_matrix,#self.t_variables['doc_l']: doc_l_matrix
                     self.t_variables['max_sent_l']: max_sent_l, #self.t_variables['max_doc_l']: max_doc_l,
                     #self.t_variables['mask_parser_1']: mask_parser_1, self.t_variables['mask_parser_2']: mask_parser_2,
                     self.t_variables['batch_l']: batch_size, self.t_variables['keep_prob']:self.config.keep_prob,
                     self.t_variables['verb_indicator']:verb_indicator_matrix}
        return  feed_dict

    def build(self):
        with tf.variable_scope("Embeddings"):
            self.embeddings = tf.get_variable("emb", [self.config.n_embed, self.config.d_embed], dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
            embeddings_root = tf.get_variable("emb_root", [1, 1, 2 * self.config.dim_sem], dtype=tf.float32,
                                                  initializer=tf.contrib.layers.xavier_initializer())
            embeddings_root_s = tf.get_variable("emb_root_s", [1, 1,2* self.config.dim_sem], dtype=tf.float32,
                                                    initializer=tf.contrib.layers.xavier_initializer())
        with tf.variable_scope("Model"):
            w_comb = tf.get_variable("w_comb", [4 * self.config.dim_sem, 2 * self.config.dim_sem], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            b_comb = tf.get_variable("bias_comb", [2 * self.config.dim_sem], dtype=tf.float32, initializer=tf.constant_initializer())

            w_comb_s = tf.get_variable("w_comb_s", [4 * self.config.dim_sem, 2 * self.config.dim_sem], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            b_comb_s = tf.get_variable("bias_comb_s", [2 * self.config.dim_sem], dtype=tf.float32, initializer=tf.constant_initializer())

            w_softmax = tf.get_variable("w_softmax", [2 * self.config.dim_sem, self.config.dim_output], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            b_softmax = tf.get_variable("bias_softmax", [self.config.dim_output], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())

        '''with tf.variable_scope("Structure/doc"):
            tf.get_variable("w_parser_p", [2 * self.config.dim_str, 2 * self.config.dim_str],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("w_parser_c", [2 * self.config.dim_str, 2 * self.config.dim_str],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("w_parser_s", [2 * self.config.dim_str, 2 * self.config.dim_str], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("bias_parser_p", [2 * self.config.dim_str], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("bias_parser_c", [2 * self.config.dim_str], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("w_parser_root", [2 * self.config.dim_str, 1], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())'''

        with tf.variable_scope("Structure/sent"):
            tf.get_variable("w_parser_p", [2 * self.config.dim_str, 2 * self.config.dim_str],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("w_parser_c", [2 * self.config.dim_str, 2 * self.config.dim_str],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("bias_parser_p", [2 * self.config.dim_str], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("bias_parser_c", [2 * self.config.dim_str], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())

            tf.get_variable("w_parser_s", [2 * self.config.dim_str, 2 * self.config.dim_str], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
            tf.get_variable("w_parser_root", [2 * self.config.dim_str, 1], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())

        sent_l = self.t_variables['sent_l']

        max_sent_l = self.t_variables['max_sent_l']

        batch_l = self.t_variables['batch_l']

        tokens_input = tf.nn.embedding_lookup(self.embeddings, self.t_variables['token_idxs'][:, :max_sent_l])
        tokens_input = tf.nn.dropout(tokens_input, self.t_variables['keep_prob'])
        verb_indicator = tf.expand_dims(self.t_variables['verb_indicator'],2)
        tokens_input = tf.concat([tokens_input,verb_indicator],2)

        mask_tokens = self.t_variables['mask_tokens'][:,  :max_sent_l]
        #mask_sents = self.t_variables['mask_sents'][:]
        [_, _, rnn_size] = tokens_input.get_shape().as_list()
        tokens_input_do = tf.reshape(tokens_input, [batch_l , max_sent_l, rnn_size])

        sent_l = tf.reshape(sent_l, [batch_l ])
        mask_tokens = tf.reshape(mask_tokens, [batch_l , -1])

        tokens_output, _ = dynamicBiRNN(tokens_input_do, sent_l, n_hidden=self.config.dim_hidden,
                                        cell_type=self.config.rnn_cell, cell_name='Model/sent')
        tokens_sem = tf.concat([tokens_output[0][:,:,:self.config.dim_sem], tokens_output[1][:,:,:self.config.dim_sem]], 2)
        tokens_str = tf.concat([tokens_output[0][:,:,self.config.dim_sem:], tokens_output[1][:,:,self.config.dim_sem:]], 2)
        temp1 = tf.zeros([batch_l , max_sent_l,1], tf.float32)
        temp2 = tf.zeros([batch_l ,1,max_sent_l], tf.float32)

        mask1 = tf.ones([batch_l , max_sent_l, max_sent_l-1], tf.float32)
        mask2 = tf.ones([batch_l , max_sent_l-1, max_sent_l], tf.float32)
        mask1 = tf.concat([temp1,mask1],2)
        mask2 = tf.concat([temp2,mask2],1)

        str_scores_s_ = get_structure('sent', tokens_str, max_sent_l, mask1, mask2)  # batch_l,  sent_l+1, sent_l
        str_scores_s = tf.matrix_transpose(str_scores_s_)  # soft parent
        tokens_sem_root = tf.concat([tf.tile(embeddings_root_s, [batch_l , 1, 1]), tokens_sem], 1)
        tokens_output_ = tf.matmul(str_scores_s, tokens_sem_root)
        tokens_output = LReLu(tf.tensordot(tf.concat([tokens_sem, tokens_output_], 2), w_comb_s, [[2], [0]]) + b_comb_s)
        print("tokens output shape", tokens_output.shape)


        '''f (self.config.sent_attention == 'sum'):
            tokens_output = tokens_output * tf.expand_dims(mask_tokens,2)
            tokens_output = tf.reduce_sum(tokens_output, 1)
        elif (self.config.sent_attention == 'mean'):
            tokens_output = tokens_output * tf.expand_dims(mask_tokens,2)
            tokens_output = tf.reduce_sum(tokens_output, 1)/tf.expand_dims(tf.cast(sent_l,tf.float32),1)
        elif (self.config.sent_attention == 'max'):
            tokens_output = tokens_output + tf.expand_dims((mask_tokens-1)*999,2)
            tokens_output = tf.reduce_max(tokens_output, 1)'''


        #sents_input = tf.reshape(tokens_output, [batch_l, 2*self.config.dim_sem])

        print("tokens output shape", tokens_output.shape)
        #print("sents input shape",sents_input.shape)
        ntime_steps = tf.shape(tokens_output)[1]
        context_rep_flat = tf.reshape(tokens_output, [-1, 2 * self.config.dim_sem])
        pred = tf.matmul(context_rep_flat, w_softmax) + b_softmax
        self.final_output = tf.reshape(pred, [-1, ntime_steps, self.config.dim_output])

        #final_output = MLP(tokens_output, 'output', self.t_variables['keep_prob'])
        #self.final_output = tf.matmul(tokens_output, w_softmax) + b_softmax
        print("final output shape", self.final_output.shape)


    def get_loss(self):

        if (self.config.opt == 'Adam'):
            optimizer = tf.train.AdamOptimizer(self.config.lr)
        elif (self.config.opt == 'Adagrad'):
            optimizer = tf.train.AdagradOptimizer(self.config.lr)
        elif (self.config.opt == 'SGD'):
            optimizer = tf.train.GradientDescentOptimizer(self.config.lr)
        elif (self.config.opt == 'Momentum'):
            optimizer = tf.train.MomentumOptimizer(self.config.lr,self.config.momentum)


        with tf.variable_scope("Model"):
            #self.transition_params = tf.placeholder(tf.float32, shape=(self.config.dim_output,self.config.dim_output),name='transition_params')

            self.trans_outputs = self.final_output  # tf.transpose(self.outputs, perm=[1, 0, 2], name='trans_outputs')
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(self.trans_outputs, self.t_variables['gold_labels'],
                                                                  self.t_variables['sent_l'])
            self.transition_params = transition_params
            self.loss = tf.reduce_mean(-log_likelihood)

            #gvs = optimizer.compute_gradients(self.cross_entropy)
            #capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs if var.name.find(NAMESPACE) != -1]
            #self.train_step = optimizer.apply_gradients(gvs)

            #mask = tf.sequence_mask(self.t_variables['sent_l'])
            # apply mask
            #self.loss = tf.boolean_mask(self.loss, mask)
            #self.loss = tf.reduce_mean(self.loss)
            #model_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Model')
            #str_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Structure')
            #for p in model_params + str_params:
            #    if ('bias' not in p.name):
            #        self.loss += self.config.norm * tf.nn.l2_loss(p)


            self.opt = optimizer.minimize(self.loss)



def pad_sequence(seq, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []


    seq = list(seq)
    seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
    sequence_padded +=  seq_
    sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length
