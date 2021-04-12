from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, \
    SpatialDropout1D, Dropout, Flatten, GlobalAveragePooling1D, Input, \
    Concatenate, Bidirectional, GRU
from tensorflow.keras.optimizers import Adam
from .nn import NN

class GRUCNN(NN):
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
        model.add(SpatialDropout1D(0.2))
        model.add(Bidirectional(GRU(128, return_sequences=True,dropout=0.1,recurrent_dropout=0.1)))
        model.add(Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform"))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(self.num_class, activation='sigmoid'))
        model.compile(optimizer=Adam(lr=1e-3),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        return model
