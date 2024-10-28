import random
import numpy as np
import ATV



def generate_f1(D,train_samples,r_data1, d_data1,feature_MFr, feature_MFd):
    vect_len1 = r_data1.shape[1]
    vect_len2 = d_data1.shape[1]
    train_n = train_samples.shape[0]
    train_feature = np.zeros([train_n,  vect_len1+vect_len2 +2*D ])

    train_label = np.zeros([train_n])
    for i in range(train_n):
        train_feature[i, 0:vect_len1] = r_data1[train_samples[i, 0], :]
        train_feature[i, vect_len1:vect_len1+vect_len2] = d_data1[train_samples[i, 1], :]
        train_feature[i, vect_len1+vect_len2:vect_len1+vect_len2 + D] =  feature_MFr[train_samples[i, 0], :]
        train_feature[i, vect_len1+vect_len2  +D:(vect_len1+vect_len2  + 2*D)] = feature_MFd[train_samples[i, 1],:]





        train_label[i] = train_samples[i, 2]
    return train_feature, train_label


# 交替最小二乘更新函数
def als(Y_WKNKN, S_r, S_d,SRW,SDW, A_init, B_init, lambda_1, lambda_s, lambda_m, mu, num_iters):
    A, B = A_init, B_init
    for iter in range(num_iters):
        # 更新 A
        BBT = B.T @ B
        ATV_grad = ATV.compute_expression(A)
        term_1_A = Y_WKNKN @ B + lambda_s * np.multiply(SRW,S_r) @ A + mu * ATV_grad
        term_2_A = BBT + lambda_1 * np.eye(B.shape[1]) + lambda_s * A.T @ A
        A = np.linalg.solve(term_2_A, term_1_A.T).T


        # 更新 B
        AAT = A.T @ A
        ATV_grad1 = ATV.compute_expression(B)
        term_1_B = Y_WKNKN.T @ A + lambda_m * np.multiply(SDW,S_d)  @ B + mu * ATV_grad1
        term_2_B = AAT + lambda_1 * np.eye(A.shape[1]) + lambda_m * B.T @ B


        B = np.linalg.solve(term_2_B, term_1_B.T).T

    return A, B

def run_MC_2(Y,SR,SD,SRW,SDW):
    # 初始化参数
    Y_WKNKN = Y
    S_r = SR
    S_d = SD
    lambda_1 = 3
    lambda_s = 5
    lambda_m = 20
    mu = 0.01
    num_iters = 100

    #SVD

    U, S, V = np.linalg.svd(Y)
    S=np.sqrt(S)
    r = 50
    Wt = np.zeros([r,r])
    for i in range(0,r):
        Wt[i][i]=S[i]
    U= U[:, 0:r]
    V= V[0:r,:]
    A = np.dot(U,Wt)
    B1 = np.dot(Wt,V)
    B=np.transpose(B1)
    A_init= A
    B_init = B

    # 运行交替最小二乘法
    A_opt, B_opt = als(Y_WKNKN, S_r, S_d,SRW,SDW, A_init, B_init, lambda_1, lambda_s, lambda_m, mu, num_iters)


    return  A_opt, B_opt
