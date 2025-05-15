
import numpy as np
import tensorflow as tf
from keras import layers, models, regularizers  # 添加正则化
from sklearn.ensemble import RandomForestClassifier
from numpy import matlib as nm
import pandas as pd
from sklearn import preprocessing
from keras import utils
import random
# 设置随机数种子
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)  # 在代码运行时设置随机数种子

def custom_loss(y_true, y_pred, encoder, lambda_l1=0.4, lambda_nuclear=0.6):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    # 初始化正则化损失
    l1_loss = 0
    nuclear_norm_loss = 0

    for layer in encoder.layers:
        if isinstance(layer, layers.Dense):  # 确保是 Dense 层
            encoder_weights = layer.weights[0]  # 获取权重

    # 计算L1正则化损失
    l1_loss += tf.reduce_sum(tf.abs(encoder_weights))

    # 计算核范数正则化损失
    # SVD分解获取奇异值
    singular_values = tf.linalg.svd(encoder_weights, compute_uv=False)
    nuclear_norm_loss += tf.reduce_sum(singular_values)

    # 总损失 = L1正则化 + 核范数正则化
    total_loss =   mse_loss+ lambda_l1 * l1_loss + lambda_nuclear * nuclear_norm_loss
    return total_loss


def disease_auto_encoder( y_train):
    encoding_dim = 32
    input_vector = layers.Input(shape=(867,))

    # encoder layer
    encoded = layers.Dense(512, activation='relu')(input_vector)
    encoded = layers.Dense(128, activation='relu')(encoded)
    disease_encoder_output = layers.Dense(encoding_dim)(encoded)

    # decoder layer
    decoded = layers.Dense(128, activation='relu')(disease_encoder_output)
    decoded = layers.Dense(512, activation='relu')(decoded)
    decoded = layers.Dense(867, activation='tanh')(decoded)

    # build a autoencoder model
    # 这是解码
    autoencoder = models.Model(inputs=input_vector, outputs=decoded)

    # 这是编码
    encoder = models.Model(inputs=input_vector, outputs=disease_encoder_output)

    # 使用自定义损失函数，添加L1正则化和核范数正则化
    # autoencoder.compile(optimizer='adam',
    #                     loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, encoder))
    autoencoder.compile(optimizer='adam',
                        loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, encoder))

    autoencoder.fit(y_train, y_train, epochs=30, batch_size=50, shuffle=True)

    # 提取编码后的特征
    disease_encoded_vector = encoder.predict(y_train)
    return disease_encoded_vector



def deep_AE(MVP_association, r_sim,d_sim):
    mtrain, dtrain = data_process(MVP_association,r_sim, d_sim)
    r_features = disease_auto_encoder(mtrain)
    d_features = disease_auto_encoder(dtrain)
    return r_features, d_features


def data_process(MVP_association,r_sim, d_sim):
    sr = r_sim
    train1 = np.concatenate((MVP_association, sr ), axis=1)


    sd = d_sim

    train2 = np.concatenate((MVP_association, sd), axis=0)

    train2 = np.transpose(train2)

    return train1, train2



