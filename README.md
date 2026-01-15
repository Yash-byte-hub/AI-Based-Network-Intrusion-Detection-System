# 🛡️ AI-Based Network Intrusion Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/ML-Random%20Forest-orange)
![Groq](https://img.shields.io/badge/AI-Groq-green)

## 📌 Project Overview

This project implements an **AI-Based Network Intrusion Detection System (NIDS)** leveraging Machine Learning techniques to secure network environments. The system analyzes network traffic data to classify it as either **Benign (Safe)** or **Malicious (Attack)**.

The core detection engine is built using a **Random Forest** algorithm. To enhance user understanding, **Groq AI** is integrated to provide natural language explanations for *why* a specific packet was flagged as safe or suspicious. The application is fully interactive, built with **Streamlit**, and deployed on Hugging Face Spaces.

## 🚀 Live Deployment

The project is live and accessible without any local setup. You can train the model, simulate traffic, and view AI explanations directly in your browser.

👉 **Live Application Link:** [Click here to view on Hugging Face](https://huggingface.co/spaces/yash23140/AI_Based_Network_Intrusion_Detection_System)

### Key Features
* **Model Training:** Train the intrusion detection model on the CIC-IDS2017 dataset.
* **Live Simulation:** Simulate live network packets from test data.
* **Real-time Detection:** Instant classification of traffic as Safe or Malicious.
* **AI Analyst:** Generates human-readable explanations using Groq AI.

---

## 🧠 Technologies Used

* **Language:** Python
* **Interface:** Streamlit
* **Machine Learning:** Scikit-learn (Random Forest Algorithm)
* **Data Processing:** Pandas, NumPy
* **Dataset:** CIC-IDS2017
* **GenAI Integration:** Groq AI (LLM for explanation generation)

---

## 📂 Dataset Details

This project utilizes a subset of the **CIC-IDS2017 dataset**, specifically the `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` file.

> **Note:** Due to GitHub file size limitations, the dataset is not hosted directly in this repository. It is uploaded and utilized within the Hugging Face deployment environment for training and testing.

---

## ⚙️ How It Works

1.  **Data Ingestion:** The system loads and preprocesses the network traffic dataset.
2.  **Model Training:** A Random Forest classifier is trained to recognize patterns distinguishing benign traffic from DDoS attacks.
3.  **Simulation:** A random packet is selected from the test set to simulate "live" incoming traffic.
4.  **Prediction:** The trained model predicts the class (Benign vs. Malicious).
5.  **Explanation:** Groq AI analyzes the packet's features and the model's prediction to generate a clear, non-technical explanation.

---

## 📊 Results & Screenshots

### 1. Model Training
The model is trained using the CIC-IDS dataset, achieving high accuracy in distinguishing traffic patterns.
![Training Complete](path/to/your/training-screenshot.png)
*(Replace the link above with your actual screenshot of the training success message)*

### 2. Threat Detection
The system successfully classifies the simulated packet.
![Detection Result](path/to/your/detection-screenshot.png)
*(Replace the link above with your actual screenshot of the Safe/Malicious result)*

### 3. AI Analyst Explanation (Groq)
Groq AI breaks down the technical decision into simple language.
 ![Image Alt]([image_url](https://github.com/Yash-byte-hub/AI-Based-Network-Intrusion-Detection-System/blob/e97b61ee52f0172ac64908244919738d4c182dcd/model_traininf_complete.png))

---

## 🖥️ Local Installation (Optional)

If you wish to run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

> **⚠️ Important:** A generic `app.py` structure is assumed. You must also have a valid **Groq API Key** to enable the AI explanation feature locally.

---

## 🎓 Academic Context

This project was developed as part of the **VOIS Internship – Major Project** under the Cybersecurity domain. It serves as a practical demonstration of applying Machine Learning and Large Language Models (LLMs) to solve critical network security challenges.

## 👤 Author

**Yash Bangar**
* B.E. Electronics & Telecommunication Engineering (ENTC) Student
* VOIS Internship Participant
