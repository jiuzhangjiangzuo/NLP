from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Flatten, MaxPooling1D, Input, Concatenate
from .nn import NN

class TextCNN(NN):
    def _build(self, pretrained_embedding):
        model = Sequential()
        if pretrained_embedding is not None:
            model.add(Embedding(self.config['vocab_size'], self.config['embedding_dim'],
                                    weights=[pretrained_embedding],
                                    input_length=self.config['maxlen'], trainable=False))
        else:
            model.add(Embedding(self.config['vocab_size'], self.config['embedding_dim'],
                                    input_length=self.config['maxlen'],
                                    embeddings_initializer="uniform", trainable=True))
        model.add(Conv1D(128, 7, activation='relu',padding='same'))
        model.add(MaxPooling1D())
        model.add(Conv1D(256, 5, activation='relu',padding='same'))
        model.add(MaxPooling1D())
        model.add(Conv1D(512, 3, activation='relu',padding='same'))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_class, activation=None))
        model.add(Dense(self.num_class, activation='sigmoid'))
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        return model
