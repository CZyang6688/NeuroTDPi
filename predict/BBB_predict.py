import numpy as np
from keras.models import load_model
from sklearn import metrics
import pandas as pd

#加载数据集
df_x_test =pd.read_csv("data/BBB_test-x.csv", na_values=["?", "NA"])
df_y_test = pd.read_csv("data/BBB_test-y.csv", na_values=["?", "NA"])
df_test = pd.concat([df_x_test, df_y_test], axis=1)
clean_data_test = df_test.dropna()
x_test = clean_data_test.iloc[:, :-1].to_numpy()
y_test = clean_data_test.iloc[:, -1].to_numpy()

# 加载模型
model = load_model('model/model_BBB.h5')

# 使用模型预测测试集
y_pred = model.predict(x_test)
y_pred_binary = np.round(y_pred).astype(int)

#保存y_test和y_pred，用于以后绘制合并ROC图
#np.save("NA_y_test.npy", y_test)
#np.save("NA_y_pred.npy", y_pred)

# 计算混淆矩阵
tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred_binary).ravel()

# 计算查准率、查全率
precision = tp / (tp + fp)
recall = tp / (tp + fn)

# 计算敏感度和特异度
se = tp / (tp + fn)
sp = tn / (tn + fp)

# 计算整体预测准确度
Q = (tp + tn) / (tp + tn + fp + fn)

# 计算马修斯相关系数
C = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

# 计算 AUC
auc = metrics.roc_auc_score(y_test, y_pred)

# 计算 F-召回率
f_measure = 2 * (precision * recall) / (precision + recall)

# 输出指标
print("ACC:", Q)
print("AUC:", auc)
print("Sensitivity (SE):", se)
print("Specificity (SP):", sp)
print("MCC:", C)

print("F1-SCORE:", f_measure)
