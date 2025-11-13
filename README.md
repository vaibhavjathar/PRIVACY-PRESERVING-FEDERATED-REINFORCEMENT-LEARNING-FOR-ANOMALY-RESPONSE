# Privacy-Preserving Federated Reinforcement Learning for Anomaly Response

**Authors:**  
- Utkarsh Singh (20223296)  
- Vaibhav Jathar (20223297)

**Guide:** Dr. Shailendra Shukla  
**Institute:** Motilal Nehru National Institute of Technology Allahabad  
**Year:** 2025  

---

##  Project Overview

This repository implements a **Privacy-Preserving Federated Reinforcement Learning (PP-FedRL)** framework for network anomaly detection and automated response. The system enables multiple distributed network clients to collaboratively learn an optimal threat-response policy **without sharing raw traffic data**.

The framework integrates:

- **Federated Learning (FedAvg)** → decentralized collaborative model training  
- **RL-inspired policy updates** → actions: allow, rate-limit, block  
- **Differential Privacy (DP)** → Gaussian noise added to protect client data  
- **Non-IID data simulation** using the NSL-KDD intrusion detection dataset  

This project demonstrates how RL, FL, and DP can be combined to achieve secure, privacy-conscious cybersecurity systems.

---

##  Key Features

- Decentralized training across **3 simulated client sites**
- No raw data leaves any client (privacy-friendly)
- Lightweight neural network policy for anomaly response
- Differential Privacy noise sweep (σ = 0.0 → 1.0)
- **Utility vs DP Noise** performance curve
- Site-transfer generalization evaluation
- Clean modular architecture for reproducibility

---

## Experimental Results (Summary)

### **Centralized Baseline**
- **F1-score:** 0.9972  
- **ROC-AUC:** 0.50  

### **Federated (No DP)**
- ROC-AUC: 0.50  

### **Federated + Differential Privacy (DP Noise Sweep)**

| DP Noise σ | F1-Score | ROC-AUC |
|------------|----------|---------|
| 0.0        | 0.0000   | 0.5000 |
| 0.1        | 0.5288   | 0.5661 |
| 0.3        | 0.4927   | 0.5823 ← best |
| 0.6        | 0.4237   | 0.5735 |
| 1.0        | 0.4238   | 0.5817 |

### **Site-Transfer Result**
- **F1:** 0.9359  
- **ROC-AUC:** 0.50  

---

##  Utility vs Differential Privacy Noise Curve

The file contains the plot showing ROC-AUC vs DP noise multiplier:
![abcd](https://github.com/user-attachments/assets/30fd5796-06f6-4ae3-b20e-a9bede4c4b71)

σ = 0.3 gives best trade-off between privacy and utility.

##  Folder Structure
fedrl-anomaly/
│
├── clients/
│ └── client.py
│
├── server/
│ └── server.py
│
├── env/
│ └── network_env.py
│
├── models/
│ └── policy.py
│
├── data/
│ └── nsl-kdd_dataset/
│ └── nsl-kdd/
│ ├── KDDTrain+.txt
│ ├── KDDTest+.txt
│ ├── client_0.npz
│ ├── client_1.npz
│ └── client_2.npz
│
├── results/
│ ├── central_mlp.joblib
│ ├── experiment_results.npz
│ ├── global_fed_noisy_*.npy
│ ├── global_fed_nonprivate.npy
│ ├── global_fed_site_transfer.npy
│ └── scaler.joblib
│
├── figures/
│ └── utility_privacy_curve.png
│
├── docs/
│ └── (project documentation)
│
├── run_experiments.ipynb
└── README.md


---

##  Installation & Setup

### **Run on Google Colab (Recommended)**

1. Upload the notebook:  
   `run_experiments.ipynb`

2. Install required libraries:
bash
!pip install numpy pandas matplotlib scikit-learn joblib

3. Upload NSL-KDD files:
KDDTrain+.txt
KDDTest+.txt

4. Run notebook cells sequentially.

5. Run Locally:
    git clone <your-repo-url>
    cd fedrl-anomaly
    pip install -r requirements.txt
   
6. Place NSL-KDD files in:
      data/nsl-kdd_dataset/nsl-kdd/

7. Then open:
      jupyter notebook run_experiments.ipynb

## How It Works (Technical Summary)
1. Local Policy Training
    Each client:
    Loads its NSL-KDD subset
    Trains a lightweight RL-inspired MLP
    Computes weight updates ΔW
    Clips gradients (sensitivity control)
    Adds Gaussian noise (DP)
    Sends noisy updates to server

2. Server Aggregation
    The server:
    Receives all noisy ΔW
    Applies Federated Averaging (FedAvg)
    Produces updated global model
    Redistributes global model

3. Differential Privacy Implementation
    Noise formula:
    ΔW_noisy = Clip(ΔW, C) + N(0, σ² * C²)
   
    Where:
    C = clipping norm
    σ = noise multiplier

## Experiment Pipeline

  1. Load and normalize NSL-KDD dataset
  2. Split into 3 non-IID client subsets
  3. Train centralized baseline
  4. Run Federated Learning (5 rounds)
  5. Run Fed + DP for noise multipliers
  6. Generate utility–privacy curve
  7. Perform site-transfer evaluation
  8. Save all results to results

## Requirements
  1. numpy
  2. pandas
  3. matplotlib
  4. scikit-learn
  5. joblib

## Troubleshooting
❗ Notebook reset while generating Word file
Avoid exporting .docx inside Colab in a single step. Instead, export text and download manually.

❗ NSL-KDD upload fails in Colab
Upload files one-by-one. If still failing, mount Google Drive and copy them.

❗ ROC-AUC remains ~0.50
Expected due to:
Class imbalance
RL → classification mapping
Use F1 and qualitative patterns for analysis.

## References (IEEE Style)

1. H. B. McMahan et al., “Communication-Efficient Learning of Deep Networks from Decentralized Data,” AISTATS, 2017.
2. M. Abadi et al., “Deep Learning with Differential Privacy,” ACM CCS, 2016.
3. C. Dwork, “Differential Privacy,” ICALP, 2006.
4. NSL-KDD Dataset, Canadian Institute for Cybersecurity, 2015.
5. C. Xu et al., “Federated Reinforcement Learning for Edge Intelligence,” IEEE IoT Journal, 2021.


## ⭐ Acknowledgments

We thank MNNIT Allahabad, our guide Dr. Shailendra Shukla, and everyone who supported us through the research and experimentation.


   



