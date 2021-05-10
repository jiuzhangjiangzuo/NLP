import yaml
import logging
import argparse
from module import Preprocessor, Trainer, Predictor

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
            preprocessor = Preprocessor(config['preprocessing'], logger)
            if config['preprocessing']['input_convertor'] == 'tf_dataset':
                train_ds, validate_ds, validate_x_ds, validate_y, test_x = preprocessor.process()
                trainer = Trainer(config['training'], logger, preprocessor.classes, None, train_ds)
                if not config['training']['predict_only']:
                    model, accuracy, cls_report, history = trainer.fit_and_validate_with_tf_dataset(train_ds, validate_ds, validate_x_ds, validate_y)
            else:
                _, _, train_x, train_y, validate_x, validate_y, test_x = preprocessor.process()
                if config['training']['model_name'] not in ['naivebayse', 'smallbert']:
                    config['training']['vocab_size'] = len(preprocessor.word2ind.keys())

                pretrained_embedding = preprocessor.embedding_matrix if config['preprocessing'].get('pretrained_embedding', None) else None
                trainer = Trainer(config['training'], logger, preprocessor.classes, pretrained_embedding)
                if not config['training']['predict_only']:
                    model, accuracy, cls_report, history = trainer.fit_and_validate(train_x, train_y, validate_x, validate_y)
            if not config['training']['predict_only']:
                logger.info("accuracy:{}".format(accuracy))
                logger.info("\n{}\n".format(cls_report))
                trainer.save()
            else:
                model = trainer.load()
            predictor = Predictor(config['predict'], logger, model)
            if config['predict']['enable_calibration']:
                predictor.train_calibrators(validate_x, validate_y)
            if config['predict'].get('debug_validation', False):
                predictor.debug_validation_set(validate_x, validate_y)
            probs = predictor.predict_prob(test_x)
            predictor.save_result(preprocessor.test_ids, probs)
            logger.info("predict completed!")
        except yaml.YAMLError as err:
            logger.warning('Config file err: {}'.format(err))
