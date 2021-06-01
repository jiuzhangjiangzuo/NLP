import sys
import yaml
import keras
import pickle
import logging
import argparse
import numpy as np
from model import TextCNN, biLSTM
from keras.utils import to_categorical

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process commandline')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--log_level', type=str, default="INFO")
    args = parser.parse_args()

    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level = args.log_level)
    logger = logging.getLogger('global_logger')
    with open(args.config, 'r') as config_file:
        try:
            config = yaml.safe_load(config_file)
            cls_data = favorite_color = pickle.load(open(config['preprocessing']['csl_data_file'], "rb"))
            weather_seq_data = favorite_color = pickle.load(open(config['preprocessing']['weather_seq_data_file'], "rb"))

            config['training']['vocab_size'] = len(cls_data['word2ind'].keys())
            classifier = TextCNN(cls_data['label_list'], config['training'], 'classifier')
            seq_parser = biLSTM(weather_seq_data['label_list'], config['training'], 'seq_parser')

            classifier.load()
            seq_parser.load()

            cls_word2ind = cls_data['word2ind']
            cls_label_list = cls_data['label_list']
            seq_word2ind = weather_seq_data['word2ind']
            seq_label_list = weather_seq_data['label_list']
            weather_idx = cls_label_list.index('GetWeather')

            while True:
                query = sys.stdin.readline()
                tokens = query.strip().split()
                token_ids = [cls_word2ind[token] if token in cls_word2ind else cls_word2ind['<unk>'] for token in tokens ]
                input = np.array([token_ids], dtype=object)
                input = keras.preprocessing.sequence.pad_sequences(input, maxlen=64, padding='post',value=cls_word2ind['<pad>'])
                cls_predictions = classifier.predict_prob(input)
                pred_cls = np.argmax(cls_predictions[0])
                print("pred_cls:{}".format(cls_label_list[pred_cls]))
                if weather_idx == pred_cls:
                    print("It is Weather Query")
                else:
                    logger.info('Query not supported')

        except yaml.YAMLError as err:
            logger.warning('Config file err: {}'.format(err))
