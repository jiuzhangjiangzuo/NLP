import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# https://keras.io/examples/nlp/text_classification_with_transformer/

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                "embedding dimension = {} should be divisible by number of heads = {}".format(embed_dim, num_heads)
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, pretrained_embedding = None):
        super(TokenAndPositionEmbedding, self).__init__()
        if pretrained_embedding is not None:
            self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, weights=[pretrained_embedding], input_length=maxlen, trainable=False)
        else:
            self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=maxlen, embeddings_initializer="uniform", trainable=True)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class TransformerClassifier(object):
    def __init__(self, classes, config, pretrained_embedding):
        self.models = {}
        self.classes = classes
        self.num_class = len(classes)
        self.config = config
        self.model = self._build(pretrained_embedding)

    def _build(self, pretrained_embedding):
        embedding_dim = self.config['embedding_dim']
        num_heads = self.config['num_heads']
        ff_dim = self.config['ff_dim']
        maxlen = self.config['maxlen']
        vocab_size = self.config['vocab_size']
        drop_out_rate = self.config['dropout_rate']

        inputs = layers.Input(shape=(maxlen,))
        embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embedding_dim, pretrained_embedding)
        x = embedding_layer(inputs)
        transformer_block = TransformerBlock(embedding_dim, num_heads, ff_dim)
        x = transformer_block(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(drop_out_rate)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(drop_out_rate)(x)
        x = layers.Dense(self.num_class, activation=None)(x)
        outputs = layers.Dense(self.num_class, activation='sigmoid')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        return model

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
