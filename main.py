from data_structure import DataSet
import tensorflow as tf
import numpy as np
import pickle
import logging
from models import  StructureModel
from model_crf import  StructureModelCRF
from baseline_model import BilstmSoftmax
from baseline_crf_model import BilstmCRF
import tqdm

def load_data(config):
    with open(config.data_file, 'rb') as f:
        train, dev, test, embeddings, vocab, tag2id = pickle.load(f)
    trainset, devset, testset = DataSet(train), DataSet(dev), DataSet(test)
    #vocab = dict([(v.index,k) for k,v in vocab.items()])
    trainset.sort()
    print(trainset)
    train_batches = trainset.get_batches(config.batch_size, config.epochs, rand=True)
    dev_batches = devset.get_batches(config.batch_size, 1, rand=False)
    test_batches = testset.get_batches(config.batch_size, 1, rand=False)
    dev_batches = [i for i in dev_batches]
    test_batches = [i for i in test_batches]
    return len(train), train_batches, dev_batches, test_batches, embeddings, vocab, tag2id

def evaluate(sess, model, test_batches,path,id2tag):
    f = open(path, "w")
    corr_count, all_count = 0, 0
    for ct, batch in test_batches:
        feed_dict = model.get_feed_dict(batch,id2tag)
        feed_dict[model.t_variables['keep_prob']] = 1
        predictions = sess.run(model.final_output, feed_dict=feed_dict)
        predictions = np.argmax(predictions, 2)
        corr_count += np.sum(predictions == feed_dict[model.t_variables['gold_labels']])
        all_count += len(batch)
        for sentence in range(len(batch)):
            for token_idx in range(predictions.shape[1]):
                pred_label = id2tag[predictions[sentence,token_idx]].upper()
                actual_label = id2tag[feed_dict[model.t_variables['gold_labels']][sentence,token_idx]].upper()
                if actual_label.lower() == 'pad' : continue
                f.write( " ".join(["-1","-1","-1",actual_label,pred_label]))
                f.write("\n")
            f.write("\n")
    acc_test = 1.0 * corr_count / all_count
    f.close()
    return  acc_test

def evaluate_crf(sess, model, test_batches,path,id2tag):
    f = open(path, "w")
    corr_count, all_count = 0, 0
    for ct, batch in test_batches:
        feed_dict = model.get_feed_dict(batch,id2tag)
        feed_dict[model.t_variables['keep_prob']] = 1
        predictions,transition_params = sess.run([model.final_output,model.transition_params], feed_dict=feed_dict)
        #predictions = np.argmax(predictions, 2)
        #print(predictions)
        viterbi_sequences = []
        for i in range(predictions.shape[0]):
            logit, sequence_length  = predictions[i], feed_dict[model.t_variables['sent_l']][i]
            #print(model.t_variables['sent_l'])

            logit = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                logit, transition_params)
            viterbi_sequences += [viterbi_seq]

        #corr_count += np.sum(predictions == feed_dict[model.t_variables['gold_labels']])
        #all_count += len(batch)
        for sentence in range(len(batch)):
            for token_idx in range(len(viterbi_sequences[sentence])):
                #print(viterbi_sequences[sentence][token_idx],id2tag)
                pred_label = id2tag[viterbi_sequences[sentence][token_idx]].upper()
                actual_label = id2tag[feed_dict[model.t_variables['gold_labels']][sentence,token_idx]].upper()
                if actual_label.lower() == 'pad' : continue
                f.write( " ".join(["-1","-1","-1",actual_label,pred_label]))
                f.write("\n")
            f.write("\n")
    #acc_test = 1.0 * corr_count / all_count
    acc_test = 0
    f.close()
    return  acc_test



def run(config):
    import random

    hash = random.getrandbits(32)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ah = logging.FileHandler(str(hash)+'.log')
    ah.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    ah.setFormatter(formatter)
    logger.addHandler(ah)

    num_examples, train_batches, dev_batches, test_batches, embedding_matrix, vocab, tag2id = load_data(config)
    id2tag = {v:k for k,v in tag2id.items()}
    print(embedding_matrix.shape)
    config.n_embed, config.d_embed = embedding_matrix.shape

    config.dim_hidden = config.dim_sem+config.dim_str

    config.dim_output = len(tag2id)

    print(config.__flags)
    logger.critical(str(config.__flags))

    if config.arch == 'sa' :model = StructureModel(config)
    elif config.arch == 'bs': model = BilstmSoftmax(config)
    elif config.arch == 'bs-crf': model = BilstmCRF(config)
    elif config.arch == 'sa-crf': model = StructureModelCRF(config)

    model.build()
    model.get_loss()
    # trainer = Trainer(config)

    num_batches_per_epoch = int(num_examples / config.batch_size)
    num_steps = config.epochs * num_batches_per_epoch


    with tf.Session() as sess:
        gvi = tf.global_variables_initializer()
        sess.run(gvi)
        sess.run(model.embeddings.assign(embedding_matrix.astype(np.float32)))
        loss = 0

        for ct, batch in tqdm.tqdm(train_batches, total=num_steps):
            feed_dict = model.get_feed_dict(batch,id2tag)
            outputs,_,_loss = sess.run([model.final_output, model.opt, model.loss], feed_dict=feed_dict)
            loss+=_loss
            if(ct%config.log_period==0):
                if config.arch in  ['bs-crf','sa-crf']:
                    acc_test = evaluate_crf(sess, model, test_batches, config.test_output, id2tag)
                    acc_dev = evaluate_crf(sess, model, dev_batches, config.dev_output, id2tag)
                else:
                    acc_test = evaluate(sess, model, test_batches,config.test_output,id2tag)
                    acc_dev = evaluate(sess, model, dev_batches,config.dev_output,id2tag)
                print('Step: {} Loss: {}\n'.format(ct, loss))
                print('Test ACC: {}\n'.format(acc_test))
                print('Dev  ACC: {}\n'.format(acc_dev))
                logger.debug('Step: {} Loss: {}\n'.format(ct, loss))
                logger.debug('Test ACC: {}\n'.format(acc_test))
                logger.debug('Dev  ACC: {}\n'.format(acc_dev))
                logger.handlers[0].flush()
                loss = 0

