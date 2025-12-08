import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys
from sklearn.preprocessing import StandardScaler
import keras as k
import pandas as pd
import numpy as np


# 定义将 SMILES 转换为 MACCS 指纹的函数（去除第167位）
def smilestomaccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs = np.array(maccs)
        return maccs[:-1]  # 去除167位即最后一位
    else:
        return None


# 定义计算7个分子理化性质的函数
def calculateDescriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        alogp = Descriptors.MolLogP(mol)
        aromaticrings = Descriptors.NumAromaticRings(mol)
        hbondacceptor = Descriptors.NumHAcceptors(mol)
        hbonddonor = Descriptors.NumHDonors(mol)
        rotatablebonds = Descriptors.NumRotatableBonds(mol)
        tpsa = Descriptors.TPSA(mol)
        weight = Descriptors.MolWt(mol)
        return np.array([alogp, aromaticrings, hbondacceptor, hbonddonor, rotatablebonds, tpsa, weight])
    else:
        return None


# 初始化StandardScaler实例
scaler_BBB = StandardScaler()
scaler_NC = StandardScaler()
scaler_NT = StandardScaler()

# 读取训练数据并拟合StandardScaler
train_data_BBB = pd.read_csv("data/BBB_7.csv", header=None)
train_data_NC = pd.read_csv("data/NC_7.csv", header=None)
train_data_NT = pd.read_csv("data/NT_7.csv", header=None)

scaler_BBB.fit(train_data_BBB)
scaler_NC.fit(train_data_NC)
scaler_NT.fit(train_data_NT)


# 定义预测函数
def predictinhibitionmultimodels(smiles):
    # 计算MACCS指纹和分子理化性质
    maccs = smilestomaccs(smiles)
    descriptors = calculateDescriptors(smiles)

    if maccs is not None and descriptors is not None:
        # 标准化分子理化性质
        descriptors_BBB = scaler_BBB.transform([descriptors])
        descriptors_NC = scaler_NC.transform([descriptors])
        descriptors_NT = scaler_NT.transform([descriptors])

        # 合并MACCS指纹和标准化后的描述符
        input_BBB = np.concatenate([maccs, descriptors_BBB[0]])
        input_NC = np.concatenate([maccs, descriptors_NC[0]])
        input_NT = np.concatenate([maccs, descriptors_NT[0]])

        # 加载模型并进行预测
        model_BBB = k.models.load_model("model/model_BBB.h5")
        model_NC = k.models.load_model("model/model_NC.h5")
        model_NT = k.models.load_model("model/model_NT.h5")

        result_BBB = model_BBB.predict(np.array([input_BBB]))
        result_NC = model_NC.predict(np.array([input_NC]))
        result_NT = model_NT.predict(np.array([input_NT]))

        return result_BBB[0], result_NC[0], result_NT[0]
    else:
        return None, None, None


# 使用预测函数进行预测
smiles = "COCCCC/C(NOCCN)/C1CCC(CC1)C(F)(F)F"
results = predictinhibitionmultimodels(smiles)

# 打印每个模型的预测结果及其可能性
model_names = ["Blood Brain Barrier", "Neuronal Cytotoxicity", "Mammalian Neurotoxicity"]
for i, result in enumerate(results):
    model_name = model_names[i]
    if result is not None:
        inhibition_status = "Neurotoxicity" if result > 0.5 else "Non-Neurotoxicity"
        # 确保probability是一个浮点数
        probability = float(result[0]) if isinstance(result, np.ndarray) else float(result)
        print(f"{model_name}: {inhibition_status} (Probability: {probability:.2f})")
    else:
        print(f"{model_name}: Prediction failed due to invalid input")


