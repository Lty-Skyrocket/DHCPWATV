import numpy as np

# 参数设置
alpha = 0.5 # 重启概率
t_max = 100  # 随机游走最大步数
beta = 0.5# 重启概率
# 随机游走高阶相似性计算
def random_walk(S, t_max, alpha):
    R_t = S.copy()  # 初始相似性矩阵
    for t in range(1, t_max):
        R_t = alpha * S + (1 - alpha) * np.dot(R_t, S)
    return R_t

#药物投影空间
def fMVP_SM(Y, SM):
    m, n = Y.shape
    MVP_sm = np.zeros((m, n))
    SM_t = random_walk(SM, t_max, beta)

    for i in range(m):
        for j in range(n):  # Adjusted loop to only iterate up to i
            dot_product = np.dot(SM_t[i],Y[:, j])
            mod_j = np.linalg.norm(Y[:, j])
            MVP_sm[i,j] = dot_product / mod_j
    for i in range(m):
        for j in range(n):
            if(Y[i,j]==0):
                Y[i,j]=MVP_sm[i,j]
    MVP_sm = Y
    return MVP_sm,SM_t

#疾病投影空间
def fMVP_MM(Y, MM):
    m, n = Y.shape
    MVP_mm = np.zeros((m, n))
    MM_t = random_walk(MM, t_max, alpha)
    for i in range(m):
        for j in range(n):
            dot_product = np.dot(Y[i],MM_t[:, j])
            mod_j = np.linalg.norm(Y[i])
            MVP_mm[i,j] = dot_product / mod_j
    for i in range(m):
        for j in range(n):
            if(Y[i,j]==0):
                Y[i,j]=MVP_mm[i,j]
    MVP_mm = Y
    return MVP_mm,MM_t



def fMVP(Y,SM,MM):
    m, n = Y.shape
    MVP_association = np.zeros((m, n))
    MVP_sm,SM_t = fMVP_SM(Y, SM)
    MVP_mm, MM_t = fMVP_MM(Y, MM)
    for i in range(m):
        for j in range(n):  # Adjusted loop to only iterate up to i
            MVP_association[i,j] = ( MVP_sm[i,j]+ MVP_sm[i,j])/(np.linalg.norm(MVP_mm[i])+np.linalg.norm(MVP_mm[:, j]))

    return MVP_association,SM_t,MM_t

