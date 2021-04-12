# https://www.tensorflow.org/tutorials/text/classify_text_with_bert
import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optmizer
from .nn import NN

class SmallBert(NN):
    SMALL_BERT_HANDLE = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'
    SMALL_BERT_PREPROCSS = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

    def __init__(self, classes, config, train_ds):
        self.classes = classes
        self.num_class = len(classes)
        self.config = config
        self.model = self._build(train_ds)

    def _build(self, train_ds):
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(self.SMALL_BERT_PREPROCSS, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(self.SMALL_BERT_HANDLE, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        #net = tf.keras.layers.Dropout(self.config['dropout_rate'])(net)
        net = tf.keras.layers.Dense(self.num_class, activation=None)(net)
        net = tf.keras.layers.Dense(self.num_class, activation='sigmoid', name='classifier')(net)
        model = tf.keras.Model(text_input, net)
        optimizer = self._create_optimizer(train_ds)
        model.compile(optimizer=optimizer,
                                  loss='binary_crossentropy')
        return model

    def _create_optimizer(self, train_ds):
        steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
        num_train_steps = steps_per_epoch * self.config['epochs']
        num_warmup_steps = int(0.1*num_train_steps)

        init_lr = self.config['learning_rate']
        optimizer = optimization.create_optimizer(init_lr=init_lr,
                                                  num_train_steps=num_train_steps,
                                                  num_warmup_steps=num_warmup_steps,
                                                  optimizer_type='adamw')
        return optimizer

    def fit_and_validate_tf_dataset(self, train_ds, val_ds, val_x_ds):
        history = self.model.fit(x=train_ds, validation_data=val_ds, epochs=self.config['epochs'])
        predictions = self.predict(val_x_ds)
        return predictions, history

    def predict_prob(self, test_ds):
        return self.model.predict(test_ds)

    def predict(self, test_ds):
        probs = self.model.predict(test_ds)
        return probs >= 0.5
