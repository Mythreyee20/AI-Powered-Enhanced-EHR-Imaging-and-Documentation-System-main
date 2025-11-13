# 🏥 AI-Powered Enhanced EHR Imaging & Documentation System

## 📘 Overview
This project integrates **Electronic Health Records (EHR)**, **medical imaging**, and **AI-driven documentation** into one intelligent system.  
It automates healthcare workflows — from **data preprocessing** to **image enhancement**, **clinical summary generation**, and **predictive analytics** — improving medical efficiency, clarity, and decision-making.

---

## ⚙️ Module 1: EHR Data Integration & Preprocessing

### 🔍 Description
This module collects, cleans, and structures patient data (demographics, symptoms, lab results, diagnoses).  
It prepares a unified dataset for downstream AI analysis.

### 💡 Key Steps
- Import and preprocess raw healthcare datasets  
- Handle missing or inconsistent values  
- Normalize and format data  
- Store unified EHR data for analysis  

### 📂 Output
`healthcare_dataset.csv` – Cleaned and merged dataset  

---

## 🧠 Module 2: Medical Image Enhancement

### 🔍 Description
This module enhances diagnostic images such as **X-rays**, **CT scans**, or **MRI scans** using deep learning and image-processing filters.  
It ensures clearer visuals for accurate clinical interpretation.

### 💡 Key Steps
- Load medical images  
- Apply enhancement (contrast, noise reduction, sharpening)  
- Save enhanced results  

### 📂 Output
`Xray_enhanced/` – Folder containing improved diagnostic images  

---

## 🤖 Module 3: Intelligent Clinical Summary Generation

### 🔍 Description
This module automatically generates **concise clinical summaries** by combining EHR data, image findings, and ICD-10 codes.  
It uses an **LLM (Hugging Face model)** to write context-aware medical reports.

### 💡 Key Steps
- Read preprocessed data and image results  
- Map conditions with ICD-10 codes  
- Generate short, structured clinical summaries  
- Save reports as CSV and text files  

### 📂 Output
`Final_Clinical_Note_All.csv` – AI-generated summaries with ICD-10 mappings  

---

## 📊 Module 4: Predictive Analytics & Visualization Dashboard

### 🔍 Description
This module transforms raw and processed data into **insightful analytics and real-time dashboards** using **Streamlit** and **Matplotlib/Plotly**.  
It empowers doctors and hospitals with quick visual decision support.

### 💡 Key Features
- Predict potential diseases or risk levels based on lab values  
- Display patient history, image status, and generated reports  
- Interactive graphs for diagnosis trends, lab results, and predictions  
- Real-time EHR visualization  

### 💡 Key Steps
- Load AI-processed data and clinical notes  
- Use ML model to predict patient risk categories  
- Visualize with Streamlit (charts, filters, summary cards)  

### 📂 Output
- `Prediction_Report.csv` – Predicted outcomes and probabilities  
- Live dashboard at `http://localhost:8501`  

---

## 🧩 Tech Stack

| Category | Technologies Used |
|-----------|------------------|
| **Frontend** | Streamlit, HTML/CSS |
| **Backend** | FastAPI |
| **AI / ML** | TensorFlow, Keras, Hugging Face Transformers |
| **Data Processing** | Pandas, NumPy |
| **Image Processing** | OpenCV |
| **Visualization** | Matplotlib, Plotly |
| **Dataset** | ICD-10 Medical Dataset |

---

## 🚀 How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt

2️⃣ Run the backend
python Backend.py

3️⃣ Run the Streamlit dashboard
streamlit run Streamlit.py

📁 Folder Structure
AI-Powered-Enhanced-EHR-Imaging-and-Documentation-System
│
├── Backend.py
├── Streamlit.py
├── ehr_model/
│   ├── xray_enhancer_model.h5
│   ├── prediction_model.pkl
│
├── data/
│   ├── healthcare_dataset.csv
│   ├── Final_Clinical_Note_All.csv
│   ├── Prediction_Report.csv
│
├── Xray_enhanced/
│   └── Enhanced images
│
└── README.md

🩺 Outcomes

✅ Clean, structured EHR dataset
✅ Enhanced, high-quality diagnostic images
✅ AI-generated, structured clinical notes
✅ Interactive visualization dashboard
✅ Predictive healthcare insights
 


