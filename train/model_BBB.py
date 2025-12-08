import tensorflow as tf
from keras import metrics
import keras as k
import pandas as pd
from keras.layers import BatchNormalization, Dropout
from keras.regularizers import l1_l2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn import metrics

roundCount=100

#需要调整的参数
l11=0.0001
l22=0.0001
learning_rate=1e-4
batch_size=100
dropout=0.2


##读取训练集测试集（读之前需要手动去除表头） #MACCS指纹为166位，记得删掉167列
# 读取 CSV 文件并将非数值类型的数据转换为 NaN
df_x_train = pd.read_csv("BBB_train-x.csv", na_values=["?", "NA"])
df_y_train = pd.read_csv("BBB_train-y.csv", na_values=["?", "NA"])
df_x_test =pd.read_csv("BBB_test-x.csv", na_values=["?", "NA"])
df_y_test = pd.read_csv("BBB_test-y.csv", na_values=["?", "NA"])
# 合并 x_train 和 y_train 为一个 DataFrame
df_train = pd.concat([df_x_train, df_y_train], axis=1)
df_test = pd.concat([df_x_test, df_y_test], axis=1)
# 删除包含 NaN 值的行
clean_data_train = df_train.dropna()
clean_data_test = df_test.dropna()
# 将数据转换为数组
x_train = clean_data_train.iloc[:, :-1].to_numpy()
y_train = clean_data_train.iloc[:, -1].to_numpy()
x_test = clean_data_test.iloc[:, :-1].to_numpy()
y_test = clean_data_test.iloc[:, -1].to_numpy()

#类别权重加权平衡
# 类别加权
num_positive_samples = np.sum(y_train == 1)  # 计算正样本数量
num_negative_samples = np.sum(y_train == 0)  # 计算负样本数量
positive_weight = (num_positive_samples + num_negative_samples) / (2 * num_positive_samples)
negative_weight = (num_positive_samples + num_negative_samples) / (2 * num_negative_samples)
class_weights_dict = {0: negative_weight, 1: positive_weight}

#建模
model=k.models.Sequential()
#第一层
model.add(k.layers.Dense(500,input_dim=x_train.shape[1],activation="relu",kernel_regularizer=l1_l2(l1=l11, l2=l22),kernel_initializer='glorot_uniform',use_bias=False))
model.add(BatchNormalization())
model.add(Dropout(dropout))
#第二层
model.add(k.layers.Dense(250,input_dim=500,activation="relu",kernel_regularizer=l1_l2(l1=l11, l2=l22),kernel_initializer='glorot_uniform',use_bias=False))
model.add(BatchNormalization())
model.add(Dropout(dropout))
#第三层：
model.add(k.layers.Dense(125,input_dim=250,activation="relu",kernel_regularizer=l1_l2(l1=l11, l2=l22),kernel_initializer='glorot_uniform',use_bias=False))
model.add(BatchNormalization())
model.add(Dropout(dropout))
#第四层
model.add(k.layers.Dense(1,input_dim=125,activation="sigmoid"))


#编译模型
optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
model.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])


#训练模型
model.fit(x_train,y_train,epochs=1000,batch_size=batch_size,verbose=2, class_weight=class_weights_dict)

# 保存模型
model.save("model_BBB_.h5")


#用测试集进行预测
y_pred = model.predict(x_test)
y_pred_binary = np.round(y_pred).astype(int)

#保存y_test和y_pred，用于以后绘制合并ROC图
#np.save("y_test_model_N_1.npy", y_test)
#np.save("y_pred_model_N_2.npy", y_pred)

# 计算 SE 和 SP 指标
tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred_binary).ravel()
se = tp / (tp + fn)
sp = tn / (tn + fp)

# 计算 TP、FP、TN、FN
tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred_binary).ravel()

# 计算整体预测准确度 Q
Q = (tp + tn) / (tp + tn + fp + fn)

# 计算马修斯相关系数 C
C = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

# 计算 AUC 值
auc = roc_auc_score(y_test, y_pred)

# 计算精确率和召回率
precision = tp / (tp + fp)
recall = tp / (tp + fn)

# 计算 F-召回率
f_measure = 2 * (precision * recall) / (precision + recall)

# 绘制 ROC 曲线
#fpr, tpr, thresholds = roc_curve(y_test, y_pred)

#plt.figure()
#plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
#plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver Operating Characteristic')
#plt.legend(loc="lower right")

# 保存 ROC 曲线图像
#plt.savefig('1A2_ROC.pdf')

#评估模型
score=model.evaluate(x_test,y_test)
print('Test loss:', score[0])


print('TP:', tp)
print('FP:', fp)
print('TN:', tn)
print('FN:', fn)

print('ACC:', score[1])   #大于0.90
print('AUC:', auc)        #大于0.85
print('Sensitivity (SE):', se)   #se和sp要相差小一点，差值小于0.4，最好都比较高
print('Specificity (SP):', sp)
print('MCC:', C)    #大于0.3
print('F1-SCORE:',f_measure)   #大于0.3