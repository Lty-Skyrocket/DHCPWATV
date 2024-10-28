import numpy as np

import math
import numpy.matlib


drug_disease_M = np.loadtxt(r"association.txt", dtype=int)



#计算药物拉普拉斯核核相似性
def Laplace_drug():
    row=269

    a = 0.5

    RR1=np.matlib.zeros((row,row))

    for i in range(0,row):
        for j in range(0,row):
            RR1[i,j]=math.exp(-(1/a)*np.linalg.norm((drug_disease_M[i,]-drug_disease_M[j,])))


    RR = RR1
    return RR



#计算疾病拉普拉斯核核相似性
def Laplace_disease():
    column = 598
    a = 0.5

    DD1=np.matlib.zeros((column,column))

    for i in range(0,column):
        for j in range(0,column):
            DD1[i,j]=math.exp(-(1/a)*np.linalg.norm((drug_disease_M[:,i]-drug_disease_M[:,j])))


    DD = DD1
    return DD



def main():
    LKS_drug = Laplace_drug()
    LKS_disease = Laplace_disease()

    np.savetxt(r'LKGIP_drug.txt', LKS_drug, delimiter='\t', fmt='%.9f')
    np.savetxt(r'LKGIP_disease.txt',  LKS_disease, delimiter='\t', fmt='%.9f')

if __name__ == "__main__":

        main()