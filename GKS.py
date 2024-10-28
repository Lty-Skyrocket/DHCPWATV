import numpy as np

import math
import numpy.matlib



drug_disease_M = np.loadtxt(r"association.txt", dtype=int)




#计算药物高斯轮廓核相似性
def Gaussian_drug():
    row = 269
    sum = 0
    RR1=np.matlib.zeros((row,row))
    for i in range(0,row):
        a=np.linalg.norm(drug_disease_M[i,])*np.linalg.norm(drug_disease_M[i,])
        sum=sum+a
    ps=row/sum
    for i in range(0,row):
        for j in range(0,row):
            RR1[i,j]=math.exp(-ps*np.linalg.norm(drug_disease_M[i,]-drug_disease_M[j,])*np.linalg.norm(drug_disease_M[i,]-drug_disease_M[j,]))


    RR = RR1
    return RR



#计算疾病高斯轮廓核相似性
def Gaussian_disease():
    column = 598
    sum = 0
    DD1=np.matlib.zeros((column,column))
    for i in range(0,column):
        a=np.linalg.norm(drug_disease_M[:,i])*np.linalg.norm(drug_disease_M[:,i])
        sum=sum+a
    ps=column/sum
    for i in range(0,column):
        for j in range(0,column):
            DD1[i,j]=math.exp(-ps*np.linalg.norm(drug_disease_M[:,i]-drug_disease_M[:,j])*np.linalg.norm(drug_disease_M[:,i]-drug_disease_M[:,j]))


    DD = DD1
    return DD


def main():
    GKS_drug = Gaussian_drug()
    GKS_disease = Gaussian_disease()
    np.savetxt(r'GKGIP_drug.txt', GKS_drug, delimiter='\t', fmt='%.9f')
    np.savetxt(r'GKGIP_disease.txt',  GKS_disease, delimiter='\t', fmt='%.9f')


if __name__ == "__main__":

        main()