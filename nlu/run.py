import yaml
import pickle
import logging
import keras
import argparse
import numpy as np
import sys
import requests
import json
from model import TextCNN, biLSTM
from keras.utils import to_categorical

def execute_domain(city):
    url = "http://api.openweathermap.org/data/2.5/weather?q={}&appid=421126fbad51c268744e7cfece50779f".format(city)
    ret = requests.get(url)
    response = json.loads(ret.text)
    if response["cod"] == 200:
        return "City:{} Weather:{} Temperate:{} Humidity:{}".format(city,
            response['weather'][0]['description'], response['main']['temp'],
            response['main']['humidity'])
    else:
        return None

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
            cls_data = pickle.load(open(config['preprocessing']['csl_data_file'], "rb"))
            weather_seq_data  = pickle.load(open(config['preprocessing']['weather_seq_data_file'], "rb"))

            config['training']['vocab_size'] = len(cls_data['word2ind'].keys())
            cls_config = config['training'].copy()
            cls_config['epochs'] = config['training']['cls_epochs']
            classifier = TextCNN(cls_data['label_list'], cls_config, 'classifier')
            seq_config = config['training'].copy()
            seq_config['epochs'] = config['training']['seq_epochs']
            seq_parser = biLSTM(weather_seq_data['label_list'], seq_config, 'seq_parser')

            if not config['training']['predict_only']:
                cls_train_y = to_categorical(cls_data['train_y'])
                cls_val_y = to_categorical(cls_data['val_y'])
                predictions = classifier.fit_and_validate(cls_data['train_x'],
                    cls_train_y, cls_data['val_x'], cls_val_y)
                accuracy, cls_report = classifier.evaluate(predictions, cls_data['val_x'], cls_data['val_y'])
                logger.info("accuracy:{}".format(accuracy))
                logger.info("\n{}\n".format(cls_report))
                classifier.save()

                seq_train_y = to_categorical(weather_seq_data['train_y'])
                seq_val_y = to_categorical(weather_seq_data['val_y'])
                predictions = seq_parser.fit_and_validate(
                    weather_seq_data['train_x'], seq_train_y,
                    weather_seq_data['sample_weight'],
                    weather_seq_data['val_x'],
                    seq_val_y)
                accuracy, seq_report = seq_parser.evaluate(predictions, weather_seq_data['val_x'], weather_seq_data['val_y'])
                logger.info("accuracy:{}".format(accuracy))
                logger.info("\n{}\n".format(seq_report))
                seq_parser.save()
            else:
                classifier.load()
                seq_parser.load()

                cls_val_y = to_categorical(cls_data['val_y'])
                seq_val_y = to_categorical(weather_seq_data['val_y'])
                cls_predictions = classifier.predict(cls_data['val_x'])
                seq_predictions = seq_parser.predict(weather_seq_data['val_x'])

                accuracy, cls_report = classifier.evaluate(cls_predictions, cls_data['val_x'], cls_data['val_y'])
                logger.info("accuracy:{}".format(accuracy))
                logger.info("\n{}\n".format(cls_report))
                accuracy, seq_report = seq_parser.evaluate(seq_predictions, weather_seq_data['val_x'], weather_seq_data['val_y'])
                logger.info("accuracy:{}".format(accuracy))
                logger.info("\n{}\n".format(seq_report))


            logger.info(("=" * 20 + "\n") * 5)
            logger.info("Service is up:")
            cls_word2ind = cls_data['word2ind']
            cls_label_list = cls_data['label_list']
            seq_word2ind = weather_seq_data['word2ind']
            seq_label_list = weather_seq_data['label_list']
            weather_idx = cls_label_list.index('GetWeather')

            while True:
                #  What is the weather like in London
                query = sys.stdin.readline()
                tokens = query.strip().split()
                token_ids = [cls_word2ind[token] if token in cls_word2ind else cls_word2ind['<unk>'] for token in tokens ]
                input = np.array([token_ids], dtype=object)
                input = keras.preprocessing.sequence.pad_sequences(input, maxlen=64, padding='post',value=cls_word2ind['<pad>'])
                cls_predictions = classifier.predict_prob(input)
                pred_cls = np.argmax(cls_predictions[0])
                logger.info("pred_cls:{}".format(cls_label_list[pred_cls]))
                if weather_idx == pred_cls:
                    logger.info("It is Weather Query!")
                    token_ids = [seq_word2ind[token] if token in seq_word2ind else seq_word2ind['<unk>'] for token in tokens ]
                    input = np.array([token_ids], dtype=object)
                    input = keras.preprocessing.sequence.pad_sequences(input, maxlen=64, padding='post',value=seq_word2ind['<pad>'])
                    seq_predictions = seq_parser.predict_prob(input)
                    seq_labels_idx = np.argmax(seq_predictions[0], -1)
                    seq_labels = [seq_label_list[idx] for idx in seq_labels_idx]
                    seq_labels = seq_labels[0:len(tokens)]

                    city_name = []
                    for token, label in zip(tokens, seq_labels):
                        logger.info("Query:{} -> Label:{}".format(token, label))
                        if label == 'city':
                            city_name.append(token)

                    city_name = ' '.join(city_name)
                    if len(city_name):
                        response = execute_domain(city_name)
                        logger.info('Response:{}'.format(response))
                    else:
                        logger.info('Query not supported')
                else:
                    logger.info('Query not supported')

        except yaml.YAMLError as err:
            logger.warning('Config file err: {}'.format(err))
