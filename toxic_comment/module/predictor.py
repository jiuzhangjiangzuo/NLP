import os
import csv
import numpy as np
from .calibrator import Calibrator
from sklearn.metrics import classification_report

class Predictor(object):
    def __init__(self, config, logger, model):
        self.config = config
        self.logger = logger
        self.model = model
        if self.config['enable_calibration']:
            self.calibators = []
            for i in range(len(self.config['classes'])):
                self.calibators.append(Calibrator(model_type=self.config['calibrator_type']))

    def predict(self, test_x):
        pred_probs = self.predict_prob(test_x)
        return pred_probs >= 0.5

    def predict_raw_prob(self, test_x):
        if hasattr(self.model, 'predict_prob'):
            prob = self.model.predict_prob(test_x)
        else:
            prob = self.model.predict_proba(test_x)
        return prob

    def predict_prob(self, test_x):
        prob = self.predict_raw_prob(test_x)
        if self.config['enable_calibration']:
            prob = self._calibrate(prob)
        return prob

    def debug_validation_set(self, validate_x, validate_y):
        predictions = self.predict(validate_x)
        cls_report = classification_report(validate_y, predictions, zero_division=1)
        self.logger.info("\n{}\n".format(cls_report))

        with open("./data/debug_validation.csv", 'w') as debug_csv_file:
            header = ["pred_{}".format(cls) for cls in self.config['classes']] + self.config['classes']
            writer = csv.writer(debug_csv_file)
            writer.writerow(header)
            for pred, truth in zip(predictions.tolist(), validate_y.tolist()):
                writer.writerow(pred + truth)

    def save_result(self, test_ids, probs):
        with open(self.config['output_path'], 'w') as output_csv_file:
             header = ['id','toxic','severe_toxic','obscene','threat','insult','identity_hate']
             writer = csv.writer(output_csv_file)
             writer.writerow(header)
             for test_id, prob in zip(test_ids, probs.tolist()):
                 writer.writerow([test_id] + prob)

    def train_calibrators(self, x, y):
        self.logger.info("train calibators")
        prob = self.predict_raw_prob(x) #(batch_size, num_of_cls)
        for i in range(len(self.config['classes'])):
            category = self.config['classes'][i]
            pred_prob = prob[:, i]
            truth_label = y[:, i]
            self.calibators[i].plot_reliability_diagrams(truth_label, pred_prob, category, self.config['calibrators_output_path'])
            uncalibrated_ece, calibrated_ece = self.calibators[i].fit(truth_label, pred_prob)
            self.logger.info("class:{}, uncalibrated_ece:{} calibrated_ece:{}".format(category, uncalibrated_ece, calibrated_ece))

    def _calibrate(self, prob):
        calibrated_prob_list = []
        for i in range(len(self.config['classes'])):
            category = self.config['classes'][i]
            pred_prob = prob[:, i]
            calibrated_prob = self.calibators[i].calibrate(pred_prob)
            calibrated_prob_list.append(calibrated_prob[:, 1])
        calibrated_prob = np.stack(calibrated_prob_list, axis=1)
        return calibrated_prob
