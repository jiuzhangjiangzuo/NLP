import io
import re
import string
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

class Preprocessor(object):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.classes = self.config['classes']
        self._load_data()

    @staticmethod
    def clean_text(text):
        text = text.strip().lower().replace('\n', '')
        # tokenization
        words = re.split(r'\W+', text)  # or just words = text.split()
        # filter punctuation
        filter_table = str.maketrans('', '', string.punctuation)
        clean_words = [w.translate(filter_table) for w in words if len(w.translate(filter_table))]
        return clean_words

    def _parse(self, data_frame, is_test = False, skip_tokenization = False):
        '''
            parameters:
                data_frame
            return:
                tokenized_input (np.array)    #[i, haven, t, paraphrased, you, at, all, gary,...]
                one_hot_label (np.array)      #[0, 0, 0, 0, 0, 0]
        '''
        X = None
        if skip_tokenization:
            X = data_frame[self.config['input_text_column']].values
        else:
            X = data_frame[self.config['input_text_column']].apply(Preprocessor.clean_text).values
        Y = None
        if not is_test:
            Y = data_frame.drop([self.config['input_id_column'], self.config['input_text_column']], 1).values
        else:
            Y = data_frame.id.values
        return X, Y

    def _load_data(self):
        data_df = pd.read_csv(self.config['input_trainset'])
        data_df[self.config['input_text_column']].fillna("unknown", inplace=True)
        #data_df['none'] = 1 - data_df[self.classes].max(axis=1)
        self.data_x, self.data_y = self._parse(data_df, skip_tokenization = self.config['skip_tokenization'])
        self.train_x, self.validate_x, self.train_y, self.validate_y = train_test_split(
                        self.data_x, self.data_y,
                        test_size=self.config['split_ratio'],
                        random_state=self.config['random_seed'])

        # we don't need the added 'none' class for validation set
        # self.validate_y = np.delete(self.validate_y, -1, 1)

        test_df = pd.read_csv(self.config['input_testset'])
        test_df[self.config['input_text_column']].fillna("unknown", inplace=True)
        self.test_x, self.test_ids = self._parse(test_df, is_test=True, skip_tokenization = self.config['skip_tokenization'])

    def process(self):
        input_convertor = self.config.get('input_convertor', None)
        label_convertor = self.config.get('label_convertor', None)
        data_x, data_y, train_x, train_y, validate_x, validate_y, test_x = \
                self.data_x, self.data_y, self.train_x, self.train_y, \
                self.validate_x, self.validate_y, self.test_x

        if input_convertor == 'count_vectorization':
            train_x, validate_x, test_x= self.count_vectorization(train_x, validate_x, test_x)
            #data_x, test_x = self.count_vectorization(data_x, test_x)
        elif input_convertor == 'tfidf_vectorization':
            train_x, validate_x, test_x= self.tfidf_vectorization(train_x, validate_x, test_x)
            #data_x, test_x = self.tfidf_vectorization(data_x, test_x)
        elif input_convertor == 'nn_vectorization': # for neural network
            train_x, validate_x, test_x = self.nn_vectorization(train_x, validate_x, test_x)
            #data_x, test_x = self.nn_vectorization(data_x, test_x)
        elif input_convertor == 'tf_dataset':
            train_ds = self.tf_dataset(train_x, train_y)
            validate_ds = self.tf_dataset(validate_x, validate_y)
            validate_x_ds = self.tf_dataset(validate_x)
            test_x_ds = self.tf_dataset(test_x)
            return train_ds, validate_ds, validate_x_ds, validate_y, test_x_ds

        return data_x, data_y, train_x, train_y, validate_x, validate_y, test_x

    def count_vectorization(self, train_x, validate_x, test_x):
        vectorizer = CountVectorizer(tokenizer=lambda x:x, preprocessor=lambda x:x)
        vectorized_train_x = vectorizer.fit_transform(train_x)
        vectorized_validate_x = vectorizer.transform(validate_x)
        vectorized_test_x  = vectorizer.transform(test_x)
        return vectorized_train_x, vectorized_validate_x, vectorized_test_x

    def tfidf_vectorization(self, train_x, validate_x, test_x):
        vectorizer = TfidfVectorizer(tokenizer=lambda x:x, preprocessor=lambda x:x)
        vectorized_train_x = vectorizer.fit_transform(train_x)
        vectorized_validate_x  = vectorizer.transform(validate_x)
        vectorized_test_x  = vectorizer.transform(test_x)
        return vectorized_train_x, vectorized_validate_x, vectorized_test_x

    def nn_vectorization(self, train_x, validate_x, test_x):
        self.word2ind = {}
        self.ind2word = {}
        specialtokens = ['<pad>','<unk>']

        pretrained_embedding = self.config.get('pretrained_embedding', None)

        if pretrained_embedding is not None:
            word2embedding = Preprocessor.load_vectors(pretrained_embedding)
            vocabs = specialtokens + list(word2embedding.keys())
            vocabs_size = len(vocabs)
            self.embedding_matrix = np.zeros((vocabs_size, self.config['embedding_dim']))
            for token in specialtokens:
                word2embedding[token] = np.random.uniform(low=-1, high=1,
                                size=(self.config['embedding_dim'],))
            for idx, word in enumerate(vocabs):
                self.word2ind[word] = idx
                self.ind2word[idx] = word
                self.embedding_matrix[idx] = word2embedding[word]
        else:
            def addword(word2ind,ind2word,word):
                if word in word2ind:
                    return
                ind2word[len(word2ind)] = word
                word2ind[word] = len(word2ind)

            for token in specialtokens:
                addword(self.word2ind, self.ind2word, token)

            for sent in train_x:
                for word in sent:
                    addword(self.word2ind, self.ind2word, word)

        train_x_ids = []
        for sent in train_x:
            indsent = [self.word2ind.get(i, self.word2ind['<unk>']) for i in sent]
            train_x_ids.append(indsent)

        train_x_ids = np.array(train_x_ids)

        validate_x_ids = []
        for sent in validate_x:
            indsent = [self.word2ind.get(i, self.word2ind['<unk>']) for i in sent]
            validate_x_ids.append(indsent)

        validate_x_ids = np.array(validate_x_ids)

        test_x_ids = []
        for sent in test_x:
            indsent = [self.word2ind.get(i, self.word2ind['<unk>']) for i in sent]
            test_x_ids.append(indsent)

        test_x_ids = np.array(test_x_ids)

        train_x_ids = keras.preprocessing.sequence.pad_sequences(train_x_ids, maxlen=self.config['maxlen'], padding='post',value=self.word2ind['<pad>'])
        validate_x_ids = keras.preprocessing.sequence.pad_sequences(validate_x_ids, maxlen=self.config['maxlen'], padding='post',value=self.word2ind['<pad>'])
        test_x_ids = keras.preprocessing.sequence.pad_sequences(test_x_ids, maxlen=self.config['maxlen'], padding='post',value=self.word2ind['<pad>'])

        return train_x_ids, validate_x_ids, test_x_ids

    @staticmethod
    def load_vectors(fname):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        #n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = np.array(list(map(float, tokens[1:])))
        return data

    def tf_dataset(self, x, y=None):
        if y is not None:
            dataset = tf.data.Dataset.from_tensor_slices((x, y))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(x)
        dataset = dataset.batch(self.config['batch_size'])
        return dataset
