# 🧠 **NeuroTDPi: Interpretable deep learning models with multimodal fusion for identifying neurotoxic compounds**

We developed a multilayer fully connected deep neural network model, designated **NeuroTDPi**, using multimodal fusion of molecular characterization with representations targeting of three neurotoxicity endpoints, including:

1. **Blood-Brain Barrier (BBB) Permeability**
2. **Neuronal Cytotoxicity (NC)**
3. **Mammalian Neurotoxicity (NT)**

<div align="center">
  <img src="TOC.png" alt="NeuroTDPi Framework Overview" width="700">
  <br>
  <em>Figure 1: Schematic overview of the NeuroTDPi framework</em>
</div>

## 📁 **Project Structure**
NeuroTDPi/
├── data/ # Datasets for BBB,NC,NT endpoints
├── model/ # Pretrained models (BBB, NC, NT)
├── predict/ # Prediction scripts
├── train/ # Training scripts
├── NeuroTDPi.py # Main prediction pipeline
├── TOC.png # Graphical abstract
└── environment.yml # Conda environment configuration


## 🔨 **1. Installation**

### **Using Conda (Recommended)**

```bash
# Clone the repository
git clone https://github.com/CZyang6688/NeuroTDPi.git

# Navigate to the project directory
cd NeuroTDPi

# Create and activate conda environment
conda create -n neuron python=3.10
conda activate neuron

# Install dependencies
conda env create -f environment.yml


## 🔨 **2. Single Endpoint Results Reproduction**

**• For BBB endpoints**
```bash
python predict/BBB_predict.py

**• For NC endpoints**
```bash
python predict/NC_predict.py

**• For NT endpoints**
```bash
python predict/NT_predict.py

## 🔨 **3. Predicting Single Compound Toxicity for BBB, NC, and NT using NeuroTDPi**
• Step 1: Modify NeuroTDPi.py file：Line 85:，Modify the SMILES string
  smiles = "SMILES of the compound to predict"  # Replace with your compound SMILES
• Step 2: Run the model
  ```bash
  python NeuroTDPi.py



