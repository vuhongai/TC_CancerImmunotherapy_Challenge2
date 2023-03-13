import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


def regression_model(
        input_size=32,
        output_size=5,
        size_dense=8,
        dropout_rate=0.15,
        l1=1e-3,
        l2=1e-4,
        learning_rate=1e-3,
        mae_weight=2
):
    def custom_KL_MAE_loss(y_true, y_pred):
        y_true = tf.multiply(y_true, [1, 1, 1, 1, 0.01])
        y_pred = tf.multiply(y_pred, [1, 1, 1, 1, 0.01])
        kl = tf.keras.metrics.kl_divergence(y_true, y_pred)
        mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
        return kl + mae * mae_weight

    input_layer = Input(shape=input_size)
    x = Dense(size_dense, activation='relu', kernel_regularizer=regularizers.L1L2(l1, l2))(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(size_dense, activation='sigmoid', kernel_regularizer=regularizers.L1L2(l1, l2))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    output_layer = Dense(output_size, activation='softmax')(x)

    model = Model(input_layer, output_layer)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=custom_KL_MAE_loss,  # "kullback_leibler_divergence",
                  metrics=['mae']
                  )
    return model



