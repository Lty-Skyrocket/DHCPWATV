import numpy as np


def get_fusion_sim (k1, k2):


    sim_r1 = np.loadtxt('GKGIP_drug.txt')
    sim_r2 = np.loadtxt('LKGIP_drug.txt')


    sim_d1 = np.loadtxt('GKGIP_disease.txt')
    sim_d2 = np.loadtxt('LKGIP_disease.txt')





    r1 = new_normalization1(sim_r1)
    r2 = new_normalization1(sim_r2)


    Sr_1 = KNN_kernel1(sim_r1, k1)
    Sr_2 = KNN_kernel1(sim_r2, k1)


    Pr = Updating1(Sr_1,Sr_2, r1, r2)
    Pr_final = (Pr + Pr.T)/2


    d1 = new_normalization1(sim_d1)
    d2 = new_normalization1(sim_d2)



    Sd_1 = KNN_kernel1(sim_d1, k2)
    Sd_2 = KNN_kernel1(sim_d2, k2)

    Pd = Updating1(Sd_1,Sd_2, d1, d2)
    Pd_final = (Pd+Pd.T)/2

    return Pr_final, Pd_final


def new_normalization1 (w):
    m = w.shape[0]
    p = np.zeros([m,m])
    for i in range(m):
        for j in range(m):
            if i == j:
                p[i][j] = 1/2
            elif np.sum(w[i,:])-w[i,i]>0:
                p[i][j] = w[i,j]/(2*(np.sum(w[i,:])-w[i,i]))
    return p


def KNN_kernel1 (S, k):
    n = S.shape[0]
    S_knn = np.zeros([n,n])
    for i in range(n):
        sort_index = np.argsort(S[i,:])
        for j in sort_index[n-k:n]:
            if np.sum(S[i,sort_index[n-k:n]])>0:
                S_knn [i][j] = S[i][j] / (np.sum(S[i,sort_index[n-k:n]]))
    return S_knn


def Updating1 (S1,S2, P1,P2):
    it = 0
    P = (P1+P2)/2
    dif = 1
    while dif>0.0000001:
        it = it + 1
        P111 =np.dot (np.dot(S1,(P2)),S1.T)
        P111 = new_normalization1(P111)
        P222 =np.dot (np.dot(S2,(P1)),S2.T)
        P222 = new_normalization1(P222)

        P1 = P111
        P2 = P222

        P_New = (P1+P2)/2
        dif = np.linalg.norm(P_New-P)/np.linalg.norm(P)
        P = P_New
    print("Iter numb1", it)
    return P

k1 = 10
k2 = 10
r_fusion_sim, d_fusion_sim = get_fusion_sim(k1, k2)
np.savetxt('drug_integration.txt', r_fusion_sim, fmt='%6f', delimiter='\t')
np.savetxt('disease_integration.txt', d_fusion_sim, fmt='%6f', delimiter='\t')











