import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve,auc
from scipy import interp
from classifiers import *
from WATV import *
from metric import *
from deep_AE import *
import warnings
from sklearn.metrics import roc_curve, auc, precision_recall_curve
warnings.filterwarnings("ignore")
import DHCP
import weight


n_splits = 5
classifier_epochs = 50
m_threshold = [0.7]
epochs = [5]
fold = 0
result = np.zeros((1, 7), float)
tprs=[]
aucs=[]
all_fpr, all_tpr, all_auc = [], [], []
mean_fpr=np.linspace(0,1,100)

all_precision, all_recall, all_aupr = [], [], []
aupr_sum, time = 0, 0
for s in itertools.product(m_threshold,epochs):

        association = np.loadtxt(r"association.txt", dtype=int)
        samples = get_all_samples(association)

        r_fusion_sim = np.loadtxt(r"drug_integration.txt", dtype=float)
        d_fusion_sim = np.loadtxt(r"disease_integration.txt", dtype=float)


        SRW = weight.calculate_weight_matrix(r_fusion_sim, 70)
        SDW = weight.calculate_weight_matrix(d_fusion_sim, 70)


        kf = KFold(n_splits=n_splits, shuffle=True, random_state=9999)

        # drug and disease features extraction from NMF
        D = 50


        WATV_rfeature, WATV_dfeature = run_MC_2(association, r_fusion_sim, d_fusion_sim,SRW,SDW)

        for train_index, val_index in kf.split(samples):
            fold += 1
            train_samples = samples[train_index, :]
            val_samples = samples[val_index, :]
            new_association = association.copy()
            for i in val_samples:
                new_association[i[0], i[1]] = 0



            #网络空间一致性投影

            MVP_association, r_sim, d_sim = DHCP.fMVP(new_association, r_fusion_sim, d_fusion_sim)



            r_features, d_features = deep_AE(MVP_association, r_sim, d_sim)

            # get feature and label
            train_feature, train_label = generate_f1(D, train_samples, r_features,d_features, WATV_rfeature, WATV_dfeature)
            val_feature, val_label = generate_f1(D, val_samples, r_features, d_features,   WATV_rfeature, WATV_dfeature)

            # MLP classfier
            model = BuildModel(train_feature, train_label)
            test_N = val_samples.shape[0]
            y_score = np.zeros(test_N)
            y_score = model.predict(val_feature)[:, 0]

            # calculate metrics
            # fpr, tpr, thresholds = roc_curve(val_label, y_score)
            # tprs.append(interp(mean_fpr, fpr, tpr))
            # tprs[-1][0] = 0.0
            # roc_auc = auc(fpr, tpr)
            # aucs.append(roc_auc)
            #
            # result += get_metrics(val_label, y_score)
            # print('[aupr, auc, f1_score, accuracy, recall, specificity, precision]',
            #       get_metrics(val_label, y_score))

            # all_fpr.append(fpr)
            # all_tpr.append(tpr)
            # Calculate precision-recall curve
            precision, recall, thresholds = precision_recall_curve(val_label, y_score)

            # Interpolate the recall-precision values for consistent comparison across folds
            interp_recall = np.linspace(0, 1, 100)  # Uniformly spaced recall values
            interp_precision = np.interp(interp_recall, recall[::-1],
                                         precision[::-1])  # Reverse recall and precision for interpolation
            interp_precision[0] = 1.0  # Ensure the starting precision is 1.0 for recall = 0

            # Store interpolated values
            all_precision.append(interp_precision)
            all_recall.append(interp_recall)

            # Calculate AUPR for this fold
            aupr = auc(recall, precision)
            all_aupr.append(aupr)

            for i in range(len(recall)):
                if recall[i] == 1:
                    precision[i] = 0

            aupr = auc(recall, precision)
            aupr_sum = aupr_sum + aupr
            time += 1
            s = aupr_sum / time
            print(f'Fold {time}: AUPR = {aupr:.4f}, Cumulative AUPR = {s:.4f}')

            # 存储每个fold的precision-recall曲线
            # all_precision.append(precision)
            # all_recall.append(recall)
            # all_aupr.append(aupr)

            # all_fpr.append(fpr)
            # all_tpr.append(tpr)
            # all_auc.append(roc_auc)

            result += get_metrics(val_label, y_score)
            print('[aupr, auc, f1_score, accuracy, recall, specificity, precision]',
                  get_metrics(val_label, y_score))
        print("==================================================")
        print(result / n_splits)
        #
        # plt.figure(figsize=(8, 6))
        #
        # for i in range(len(all_fpr)):
        #     plt.plot(all_fpr[i], all_tpr[i], label=f'ROC fold {i + 1} (AUC = {all_auc[i]:.4f})', linestyle='-',
        #              linewidth=2)
        # # 先找到最小的长度
        # min_length = min(len(fpr) for fpr in all_fpr)
        #
        # # 对每个子数组进行截断或插值，使它们具有相同的长度
        # all_fpr_fixed = [np.interp(np.linspace(0, 1, min_length), fpr, fpr) for fpr in all_fpr]
        # all_tpr_fixed = [np.interp(np.linspace(0, 1, min_length), tpr, tpr) for tpr in all_tpr]
        # # 然后计算平均值
        # mean_fpr = np.mean(all_fpr_fixed, axis=0)
        # mean_tpr = np.mean(all_tpr_fixed, axis=0)
        # mean_auc = np.mean(all_auc)
        # # np.savetxt(r'mean_fpr.txt', mean_fpr, delimiter='\t', fmt='%.9f')
        # # np.savetxt(r'mean_tpr.txt', mean_tpr, delimiter='\t', fmt='%.9f')
        #
        # plt.plot(fpr, tpr, label=f'Mean ROC (AUC = {mean_auc:.4f})', linestyle='-')
        # # Plot the diagonal chance line
        # plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        # plt.xlim([-0.05, 1.05])
        # plt.ylim([-0.05, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('ROC Curve for 5-Fold CV')
        # plt.legend(loc='lower right')
        # # 保存图像到文件
        # plt.savefig('roc_5fold-duijiao.png')
        # plt.show()

        # 绘制Precision-Recall曲线
        plt.figure(figsize=(8, 6))

        for i in range(len(all_precision)):
            plt.plot(all_recall[i], all_precision[i], label=f'PR fold {i + 1} (AUPR = {all_aupr[i]:.4f})',
                     linestyle='-')

        # 先找到最小的长度
        min_length = min(len(recall) for recall in all_recall)

        # 对每个子数组进行截断或插值，使它们具有相同的长度
        all_recall_fixed = [np.interp(np.linspace(0, 1, min_length), recall, recall) for recall in all_recall]
        all_precision_fixed = [np.interp(np.linspace(0, 1, min_length), precision, precision) for precision in
                               all_precision]

        # 然后计算平均值
        mean_recall = np.mean(all_recall_fixed, axis=0)
        mean_precision = np.mean(all_precision_fixed, axis=0)
        mean_aupr = np.mean(all_aupr)

        plt.plot(recall, precision, label=f'Mean PR (AUPR = {mean_aupr:.4f})', linestyle='-')
        plt.plot([0, 1], [1, 0], linestyle='--', color='gray')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR Curve for 5-Fold CV')
        plt.legend(loc='lower left')
        # 保存图像到文件
        plt.savefig('pr_curve_5fold.png')
        plt.show()


