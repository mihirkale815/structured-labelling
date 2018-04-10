from data_structure import Corpus
import argparse


import pickle

def main(train_path, dev_path, test_path, output_path, emb_path):
    corpus = Corpus()
    corpus.load(train_path, 'train')
    corpus.load(dev_path, 'dev')
    corpus.load(test_path, 'test')
    corpus.preprocess()
    options =  dict(max_sents=60, max_tokens=100, skip_gram=False, emb_size=100, emb_path=emb_path)
    print('Start training word embeddings')
    corpus.build_vocab()
    #corpus.w2v(options)

    instance, instance_dev, instance_test, embeddings, vocab ,tag2id = corpus.prepare(options)
    pickle.dump((instance, instance_dev, instance_test, embeddings, vocab, tag2id),open(output_path,'wb'))


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('train_path', action="store")
parser.add_argument('dev_path', action="store")
parser.add_argument('test_path', action="store")
parser.add_argument('output_path', action="store")
parser.add_argument('emb_path', action="store")
args = parser.parse_args()

# train_path = '../data/yelp-2013.train'
# dev_path = '../data/yelp-2013.dev'
# test_path = '../data/yelp-2013.test'

main(args.train_path, args.dev_path, args.test_path, args.output_path, args.emb_path)