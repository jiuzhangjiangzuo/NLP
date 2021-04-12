from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from module.model import NaiveBayer, TextCNN, TextRNN, biLSTM, TransformerClassifier, GRUCNN, SmallBert

class Trainer(object):
    def __init__(self, config, logger, classes, pretrained_embedding, train_ds = None):
        self.config = config
        self.logger = logger
        self.classes = classes
        self.pretrained_embedding = pretrained_embedding
        self._create_model(classes, train_ds)

    def _create_model(self, classes, train_ds):
        if self.config['model_name'] == 'naivebayse':
            self.model = NaiveBayer(classes)
        elif self.config['model_name'] == 'textcnn':
            self.model = TextCNN(classes, self.config, self.pretrained_embedding)
        elif self.config['model_name'] == 'textrnn':
            self.model = TextRNN(classes, self.config, self.pretrained_embedding)
        elif self.config['model_name'] == 'bilstm':
            self.model = biLSTM(classes, self.config, self.pretrained_embedding)
        elif self.config['model_name'] == 'transformer':
            self.model = TransformerClassifier(classes, self.config, self.pretrained_embedding)
        elif self.config['model_name'] == 'grucnn':
            self.model = GRUCNN(classes, self.config, self.pretrained_embedding)
        elif self.config['model_name'] == 'smallbert':
            self.model = SmallBert(classes, self.config, train_ds)
        else:
            self.logger.warning("Model Type:{} is not support yet".format(self.config['model_name']))

    def metrics(self, predictions, labels):
        accuracy = accuracy_score(labels, predictions)
        cls_report = classification_report(labels, predictions, zero_division=1)
        return accuracy, cls_report

    def fit(self, train_x, train_y):
        self.model.fit(train_x, train_y)
        return self.model

    def validate(self, validate_x, validate_y):
        predictions = self.model.predict(validate_x)
        return self.metrics(predictions, validate_y)

    def fit_and_validate(self, train_x, train_y, validate_x, validate_y):
        predictions, history = self.model.fit_and_validate(train_x, train_y, validate_x, validate_y)
        accuracy, cls_report = self.metrics(predictions, validate_y)
        return self.model, accuracy, cls_report, history

    def fit_and_validate_with_tf_dataset(self, train_ds, validate_ds, validate_x_ds, validate_y):
        predictions, history = self.model.fit_and_validate_tf_dataset(train_ds, validate_ds, validate_x_ds)
        accuracy, cls_report = self.metrics(predictions, validate_y)
        return self.model, accuracy, cls_report, history

    def save(self):
        return self.model.save()

    def load(self):
        return self.model.load()
