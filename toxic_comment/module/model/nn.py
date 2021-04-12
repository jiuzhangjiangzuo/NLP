from tensorflow import keras
import os

class NN(object):
    def __init__(self, classes, config, pretrained_embedding):
        self.classes = classes
        self.num_class = len(classes)
        self.config = config
        self.model_path = os.path.join(self.config['output_path'], self.config['model_name'])
        if self.config['predict_only']:
            self.model = self.load()
        else:
            self.model = self._build(pretrained_embedding)

    def _build(self, pretrained_embedding):
        pass

    def fit_and_validate(self, train_x, train_y, validate_x, validate_y):
        history = self.model.fit(train_x, train_y,
                            epochs=self.config['epochs'],
                            verbose=True,
                            validation_data=(validate_x, validate_y),
                            batch_size=self.config['batch_size'])
        predictions = self.predict(validate_x)
        return predictions, history

    def predict_prob(self, test_x):
        return self.model.predict(test_x)

    def predict(self, test_x):
        probs = self.model.predict(test_x)
        return probs >= 0.5

    def save(self):
        self.model.save(self.model_path)

    def load(self):
        return keras.models.load_model(self.model_path)
