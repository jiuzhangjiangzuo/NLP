from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input, Concatenate, Bidirectional, TimeDistributed
from .nn import NN
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

class biLSTM(NN):
    def _build(self):
        model = Sequential()
        model.add(Embedding(self.config['vocab_size'], self.config['embedding_dim'],
                                input_length=self.config['maxlen'],
                                embeddings_initializer="uniform", trainable=True, mask_zero=True))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(TimeDistributed(Dense(self.num_class, activation='softmax')))
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        return model

    def fit_and_validate(self, train_x, train_y, sample_weight, validate_x, validate_y):
        history = self.model.fit(train_x, train_y,
                            epochs=self.config['epochs'],
                            verbose=True,
                            validation_data=(validate_x, validate_y),
                            batch_size=self.config['batch_size'], sample_weight=sample_weight)
        predictions = self.predict(validate_x)
        return predictions

    def evaluate(self, predictions, validate_x, validate_y):
        validate_x = validate_x.flatten()
        predictions = predictions.flatten()
        validate_y = validate_y.flatten()
        # mask padding
        predictions_masked = []
        validate_y_masked = []
        for x, y, pred in zip(validate_x, validate_y, predictions):
            if x == 0:
                continue
            predictions_masked.append(pred)
            validate_y_masked.append(y)
        accuracy = accuracy_score(validate_y_masked, predictions_masked)
        cls_report = classification_report(validate_y_masked, predictions_masked, zero_division=1)
        return accuracy, cls_report
