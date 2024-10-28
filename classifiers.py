import numpy as np
import random
from keras.layers import Dense, Input, dot,Dropout, BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping
import tensorflow as tf

def get_all_samples(conjunction):
    pos = []
    neg = []
    for index in range(conjunction.shape[0]):
        for col in range(conjunction.shape[1]):
            if conjunction[index, col] == 1:
                pos.append([index, col, 1])
            else:
                neg.append([index, col, 0])
    pos_len = len(pos)
    # 固定随机种子
    random.seed(666)
    new_neg = random.sample(neg, pos_len)
    samples = pos + new_neg
    samples = random.sample(samples, len(samples))
    samples = np.array(samples)
    return samples





def BuildModel(train_x, train_y):
    l = len(train_x[1])
    inputs = Input(shape=(l,))

    x = Dense(512, activation='relu')(inputs)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    # x = BatchNormalization()(x)

    # x = Dense(128, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam',  # Change to Adam
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    model.fit(train_x, train_y, epochs=1, validation_split=0.2, callbacks=[early_stopping])  # Increase epochs
    return model