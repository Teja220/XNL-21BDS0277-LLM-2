# XNL-21BDS0277-LLM-2

# 🚀 **Fine-Tuned GPT-Neo 2.7B for Question Answering**

## 📌 **Project Overview**
This project focuses on **fine-tuning GPT-Neo 2.7B** for **question answering tasks** using the **SQuAD v2 dataset**. The model has been optimized for **fast training, reduced memory usage, and efficient inference** using **DeepSpeed, LoRA, and bf16 mixed precision training**.

The fine-tuned model can be used for various **NLP applications** such as:
- Conversational AI 🤖  
- Automated Question Answering 📚  
- Information Retrieval 🔍  

---

## 📋 **Requirements for Reproducing the Project**  
To set up and run this project, the following dependencies and resources are required:

### **🔹 System Requirements**  
- A **GPU-enabled environment** (Google Colab, AWS, Local with CUDA support).
- At least **16GB VRAM recommended** for fine-tuning.

### **🔹 Frameworks & Libraries**  
- **Hugging Face Transformers** (For model loading & training)  
- **DeepSpeed** (For optimized distributed training)  
- **PyTorch** (For fine-tuning & inference)  
- **PEFT (LoRA Fine-Tuning)** (For efficient adapter-based tuning)  
- **Accelerate** (For multi-GPU and optimized training)  
- **Weights & Biases (Optional)** (For experiment tracking)

### **🔹 Dataset**  
- **SQuAD v2** (A high-quality dataset for QA tasks)
- **Dataset Reduction**: Only **2% of the original dataset** was used for **ultra-fast training**.

### **🔹 Storage & Model Access**  
- The full fine-tuned model weights are **too large for GitHub**.
- **Download the full `pytorch_model.bin`** from:  
  🔗 **[https://colab.research.google.com/drive/1vVzHQYCJvobUpegnJ0Jowp_ZBGsJWMzR?usp=sharing]** *(Replace with actual link)*  

---

## 📦 **Deliverables**
This repository contains all the necessary files for training, evaluation, and fine-tuning of GPT-Neo 2.7B.

### **📂 Project Files**
- **`preprocess.py`** → Tokenizes and processes the dataset for training.  
- **`train.py`** → Fine-tunes GPT-Neo 2.7B using DeepSpeed & LoRA.  
- **`ds_config.json`** → DeepSpeed configuration for memory-efficient training.  
- **`README.md`** → Documentation for setup, training, and model access.

### **📂 Fine-Tuned Model Files**
- **Model Configuration** (`config.json`, `tokenizer.json`, etc.).  
- **LoRA Adapter Weights** (`adapter_model.safetensors`).  
- **Tokenizer & Vocabulary** (`vocab.json`, `merges.txt`).  
- **Training Logs & Checkpoints**.  

✅ **Note:** The full model weights **(`pytorch_model.bin`) are not included** in this repo due to size limitations.  
---

## 🚀 **Project Implementation Steps**
### **1️⃣ Data Preprocessing**
- Tokenized the **SQuAD v2 dataset** with a **max sequence length of 128** for faster training.
- Used **padding & truncation** for efficient batching.

### **2️⃣ Fine-Tuning with DeepSpeed & LoRA**
- **DeepSpeed Zero-Offload (Stage 2)** was used to reduce GPU memory usage.
- **LoRA Adapters** were implemented for parameter-efficient fine-tuning.
- **Mixed Precision Training (bf16)** was enabled to speed up computations.

### **3️⃣ Model Evaluation**
- Evaluated on **2% of the SQuAD v2 validation set**.
- Measured **accuracy, loss, and perplexity**.

### **4️⃣ Model Deployment (Optional)**
- Model can be **converted to ONNX** or **served using FastAPI**.
- **Deployment-ready for real-world QA applications**.

---

## 📊 **Performance & Results**
✅ **Successfully fine-tuned GPT-Neo 2.7B on SQuAD v2**  
✅ **Integrated DeepSpeed Zero-Offload for memory efficiency**  
✅ **Used LoRA adapters for fine-tuning with minimal memory usage**  
✅ **Enabled mixed precision (bf16) for faster training**  
✅ **Fine-tuned model achieves significant improvements in QA tasks**  

---

## 📥 **Download & Usage**
### **🔹 How to Get the Fine-Tuned Model**
The full fine-tuned model is available **on external storage** due to size constraints.  
📥 **Download the full weights (`pytorch_model.bin`) from:**  
- **Google Drive:** [Download Link] *(Replace with actual link)*  
- **Hugging Face Model Hub:** [Model Repo Link] *(Replace with actual link)*  

### **🔹 Model Inference**
Once downloaded, the model can be used for inference in a Python script or API.  
- **Load the model using Hugging Face Transformers.**
- **Provide a question + context as input** and get an answer.

---

## 🏆 **Future Improvements**
🔹 **Further Fine-Tuning on Custom Datasets**  
🔹 **Hyperparameter Tuning for Optimal Performance**  
🔹 **Deployment as a Web API for Real-Time QA**  
🔹 **Comparison with other LLMs (GPT-J, BLOOM, Falcon, etc.)**  

---

📧 For questions or contributions, reach out via:

LinkedIn: https://www.linkedin.com/in/Teja220
Email: saitejaredyy@gmail.com


📧 **For questions or contributions, reach out via GitHub Issues or Email.**

Feel free to use and modify it for research and development purposes.

---

✅ **This README ensures that anyone can understand and reproduce your project!** 🚀🔥  
Let me know if you need any modifications! 😊
