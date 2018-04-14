import gensim
import numpy as np
import re
import random
import math
import unicodedata
import itertools
from utils import grouper
import sys
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors


def sentence_generator(path):
    f = open(path)
    curr_sent = []
    for line in f:
        line = line.strip("\n")
        if line == '' or line == '\t':
            yield curr_sent
            curr_sent = []
            continue
        tag,word = line.split("\t")
        tag = tag.lower()
        word = word.lower()
        curr_sent.append((word,tag))
    yield curr_sent
    f.close()

def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', unicode(s,'utf-8'))
                  if unicodedata.category(c) != 'Mn')

class RawData:
    def __init__(self):
        self.userStr = ''
        self.productStr = ''
        self.reviewText = ''
        self.goldRating = -1
        self.predictedRating = -1
        self.sent_token_lst = []


class DataSet:
    def __init__(self, data):
        self.data = data
        self.num_examples = len(self.data)

    def sort(self):
        random.shuffle(self.data)
        self.data = sorted(self.data, key=lambda x: x._sen_len())
        #self.data = sorted(self.data, key=lambda x: x._doc_len())

    def get_by_idxs(self, idxs):
        return [self.data[idx] for idx in idxs]

    def get_batches(self, batch_size, num_epochs=None, rand = True):
        num_batches_per_epoch = int(math.ceil(self.num_examples / batch_size))
        idxs = list(range(self.num_examples))
        _grouped = lambda: list(grouper(idxs, batch_size))

        if(rand):
            grouped = lambda: random.sample(_grouped(), num_batches_per_epoch)
        else:
            grouped = _grouped
        num_steps = num_epochs*num_batches_per_epoch
        batch_idx_tuples = itertools.chain.from_iterable(grouped() for _ in range(num_epochs))
        for i in range(num_steps):
            batch_idxs = tuple(i for i in next(batch_idx_tuples) if i is not None)
            batch_data = self.get_by_idxs(batch_idxs)
            yield i,batch_data


class Instance:
    def __init__(self):
        self.token_idxs = None
        self.goldLabel = -1
        self.idx = -1

    def _sen_len(self, idx=None):
        k = len(self.token_idxs)
        return k


class Corpus:
    def __init__(self):
        self.senlst = {}

    def load(self, in_path, name):
        #self.tags = ["b-actor","b-character","b-director","b-genre","b-plot","b-rating","b-ratings_average","b-review","b-song","b-title","b-trailer","b-year"]
        #self.tags += ["i-actor","i-character","i-director","i-genre","i-plot","i-rating","i-ratings_average","i-review","i-song","i-title","i-trailer","i-year"]
        #self.tags += ['o']
        #self.tag2id = {tag:i for i,tag in enumerate(self.tags)}
        self.senlst[name] = []
        gen = sentence_generator(in_path)
        for sent in gen:
            tokens = [tup[0] for tup in sent]
            tags = [tup[1] for tup in sent]
            sen = RawData()
            sen.goldRating = tags
            sen.reviewText = " ".join(tokens)
            self.senlst[name].append(sen)
    def preprocess(self):
        random.shuffle(self.senlst['train'])
        for dataset in self.senlst:
            for sen in self.senlst[dataset]:
                sen.sent = sen.reviewText#.split('<split2>')
                sen.sent = re.sub(r"[^A-Za-z0-9(),!?\'\`_]", " ",sen.sent)
                sen.sent_token_lst = sen.sent.split()
                sen.sent_token_lst = [sent_tokens for sent_tokens in sen.sent_token_lst if(len(sent_tokens)!=0)]
            self.senlst[dataset] = [sen for sen in self.senlst[dataset] if len(sen.sent_token_lst)!=0]
        self.build_tags_vocab()
        print(self.tag2id)

    def build_vocab(self):
        self.vocab = {}
        self.vocab['pad'] = 0
        self.vocab['UNK'] = 1
        for sen in self.senlst['train']:
            for token in sen.sent_token_lst:
                if(token not in self.vocab):
                    self.vocab[token] = len(self.vocab)



    def build_tags_vocab(self):
        self.tag2id = {'pad':0}
        for sen in self.senlst['train']:
            tags = sen.goldRating
            for tag in tags:
                if(tag not in self.tag2id):
                    self.tag2id[tag] = len(self.tag2id)


    '''def w2v(self, options):
        sentences = []
        for sen in self.senlst['train']:
            sentences.append(sen.sent_token_lst)
        if('dev' in self.senlst):
            for sen in self.senlst['dev']:
                sentences.append(sen.sent_token_lst)

        if(options['skip_gram']):
            self.w2v_model = gensim.models.deprecated.word2vec.Word2Vec(size=options['emb_size'], window=5, min_count=10, workers=4, sg=1)
        else:
            self.w2v_model = gensim.models.deprecated.word2vec.Word2Vec(size=options['emb_size'], window=5, min_count=10, workers=4)
        self.w2v_model.scan_vocab(sentences)  # initial survey
        rtn = self.w2v_model.scale_vocab(dry_run = True)  # trim by min_count & precalculate downsampling
        print(rtn)
        self.w2v_model.finalize_vocab()  # build tables & arrays

        #glove_file = datapath('test_glove.txt')
        #tmp_file = get_tmpfile("test_word2vec.txt")
        #from gensim.scripts.glove2word2vec import glove2word2vec
        #glove2word2vec(glove_file, tmp_file)
        #model = KeyedVectors.load_word2vec_format(tmp_file)
        #self.w2v_model.train(sentences, total_examples=self.w2v_model.corpus_count, epochs=self.w2v_model.iter)
        #self.w2v_model.train(sentences, total_examples=self.w2v_model.corpus_count, epochs=0)
        self.vocab = self.w2v_model.wv.vocab
        print('Vocab size: {}'.format(len(self.vocab)))

        # model.save('../data/w2v.data')'''

    def prepare(self, options):
        self.emb_path = options['emb_path']
        instances, instances_dev, instances_test = [],[],[]
        instances, embeddings, vocab, tag2id = self.prepare_for_training(options)
        if ('dev' in self.senlst):
            instances_dev = self.prepare_for_test(options, 'dev')
        instances_test = self.prepare_for_test( options, 'test')
        #print(instances, instances_dev, instances_test, embeddings, vocab)
        return instances, instances_dev, instances_test, embeddings, vocab, tag2id

    def prepare_notest(self, options):
        instances, instances_dev, instances_test = [],[],[]
        instances_, embeddings, vocab, tag2id = self.prepare_for_training(options)
        print(len(instances))
        for bucket in instances_:
            num_test = len(bucket) / 10
            instances_test.append(bucket[:num_test])
            instances.append(bucket[num_test:])

        return instances, instances_dev, instances_test, embeddings, vocab, tag2id


    def prepare_for_training(self, options):

        embeddings_index = {}
        f = open(self.emb_path)
        for idx,line in enumerate(f):
            if idx%100000 == 0 : print("loaded",idx,"word vectors")
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()


        instancelst = []
        embeddings = np.zeros([len(self.vocab)+1,options['emb_size']])
        for word, i in self.vocab.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embeddings[i] = embedding_vector

        #for word in self.vocab:
        #    embeddings[self.vocab[word].index] = self.w2v_model[word]
        #self.vocab['UNK'] = gensim.models.word2vec.Vocab(count=0, index=len(self.vocab))
        n_filtered = 0
        for i_sen, sen in enumerate(self.senlst['train']):
            instance = Instance()
            instance.idx = i_sen
            max_n_tokens = len(sen.sent_token_lst)

            if(max_n_tokens>options['max_tokens']):
                n_filtered += 1
                continue

            token_idxs = []
            for i in range(len(sen.sent_token_lst)):
                token = sen.sent_token_lst[i]
                if(token in self.vocab):
                    token_idxs.append(self.vocab[token])
                else:
                    token_idxs.append(self.vocab['UNK'])

            instance.token_idxs = token_idxs
            instance.goldLabel = [self.tag2id[tag] for tag in sen.goldRating]
            #print(instance.goldLabel)
            instancelst.append(instance)
        print('n_filtered in train: {}'.format(n_filtered))
        return instancelst, embeddings, self.vocab, self.tag2id

    def prepare_for_test(self, options, name):
        instancelst = []
        n_filtered = 0
        for i_sen, sen in enumerate(self.senlst[name]):
            instance = Instance()
            instance.idx = i_sen
            max_n_tokens = len(sen.sent_token_lst)
            if(max_n_tokens>options['max_tokens']):
                n_filtered += 1
                continue

            token_idxs = []
            for i in range(len(sen.sent_token_lst)):
                token  = sen.sent_token_lst[i]
                if(token in self.vocab):
                    token_idxs.append(self.vocab[token])
                else:
                    token_idxs.append(self.vocab['UNK'])

            instance.token_idxs = token_idxs
            instance.goldLabel = [self.tag2id.get(tag,0) for tag in sen.goldRating]
            include = True
            for tag  in sen.goldRating:
                if tag not in self.tag2id :
                    print("tag",tag,"not found in tag vocab!!")
                    include = False
            if include : instancelst.append(instance)
        print('n_filtered in {}: {}'.format(name, n_filtered))
        return instancelst