# **🚀 Fine-Tuned GPT-Neo 2.7B for Question Answering**  

## **📌 Project Overview**  
This project is focused on **fine-tuning GPT-Neo 2.7B** for **question-answering tasks** using the **SQuAD v2 dataset**. The goal was to optimize the model for **faster training, lower memory usage, and improved performance** using **DeepSpeed, LoRA, and mixed precision (bf16) training**.  

The fine-tuned model can be used in various NLP applications, including:  
- **Conversational AI** 🤖  
- **Automated Q&A Systems** 📚  
- **Information Retrieval** 🔍  

Since working with large-scale LLMs can be resource-intensive, this project focuses on **efficient fine-tuning techniques** to get the best results without requiring high-end hardware.  

---

## **📋 What You Need to Reproduce This Project**  

### **🔹 System Requirements**  
- A **GPU-enabled setup** (Google Colab, AWS, or a local machine with CUDA).  
- At least **16GB VRAM recommended** for fine-tuning.  

### **🔹 Libraries & Frameworks**  
To fine-tune or run the model, you’ll need:  
- **Hugging Face Transformers** – for working with GPT-Neo.  
- **DeepSpeed** – to speed up training and reduce memory usage.  
- **PyTorch** – for training and inference.  
- **LoRA (Low-Rank Adaptation)** – for more efficient fine-tuning.  
- **Accelerate** – for optimized multi-GPU training.  
- **Weights & Biases (Optional)** – for experiment tracking.  

### **🔹 Dataset Used**  
- **SQuAD v2** – a benchmark dataset for Question Answering.  
- Since training on the full dataset is resource-intensive, **only 2%** was used for ultra-fast fine-tuning.  

### **🔹 Model Weights & Storage**  
- The full **fine-tuned model weights** are **too large for GitHub**, so they are stored externally.  
- **Download `pytorch_model.bin` from:**  
  🔗 [Google Drive Link](https://colab.research.google.com/drive/1vVzHQYCJvobUpegnJ0Jowp_ZBGsJWMzR?usp=sharing)  

---

## **📦 Project Deliverables**  
This repository contains everything needed to fine-tune GPT-Neo 2.7B and reproduce the results.  

### **📂 Files in This Repo**  
- **`preprocess.py`** – Tokenizes and prepares the dataset.  
- **`train.py`** – Fine-tunes GPT-Neo 2.7B using DeepSpeed & LoRA.  
- **`ds_config.json`** – Configuration file for DeepSpeed optimization.  
- **`README.md`** – Documentation for setup and model access.  

### **📂 Model Files (Stored Externally)**  
- **Model Configurations** (`config.json`, `tokenizer.json`, etc.).  
- **LoRA Adapter Weights** (`adapter_model.safetensors`).  
- **Tokenizer & Vocabulary** (`vocab.json`, `merges.txt`).  
- **Training Logs & Checkpoints**.  
---

## **🚀 How the Model Was Trained**  

### **1️⃣ Preprocessing the Dataset**  
- The **SQuAD v2 dataset** was tokenized with a **max sequence length of 128** for faster training.  
- **Padding & truncation** were applied to ensure consistent input sizes.  

### **2️⃣ Fine-Tuning Process**  
- Used **DeepSpeed Zero-Offload (Stage 2)** to handle large model sizes efficiently.  
- **LoRA adapters** were applied to reduce the number of trainable parameters.  
- **bf16 Mixed Precision** was enabled to speed up training and reduce memory usage.  

### **3️⃣ Model Evaluation**  
- The fine-tuned model was tested on **2% of the SQuAD v2 validation set**.  
- Performance was measured using **accuracy, loss, and perplexity scores**.  

### **4️⃣ Deployment Possibilities (Optional)**  
- The model can be **converted to ONNX** for lightweight deployment.  
- It can be **hosted using FastAPI** to serve real-time responses.  

---

## **📊 Model Performance & Results**  

✅ Successfully fine-tuned **GPT-Neo 2.7B** on a subset of **SQuAD v2**  
✅ Integrated **DeepSpeed Zero-Offload** for efficient memory usage  
✅ Used **LoRA adapters** to fine-tune without retraining the full model  
✅ Enabled **mixed precision (bf16)** to speed up training  
✅ Achieved significant improvements in **Question Answering tasks**  

---

## **📥 Download & Use the Fine-Tuned Model**  

Since GitHub has file size limitations, **the full model weights must be downloaded separately**:  

📥 **Download `pytorch_model.bin` from:**  
- **Google Drive:** [Download Link](https://colab.research.google.com/drive/1vVzHQYCJvobUpegnJ0Jowp_ZBGsJWMzR?usp=sharing)  

---

## **🏆 Future Plans for the Project**  
🔹 Experimenting with **larger datasets for improved performance**  
🔹 **Tuning hyperparameters** to optimize accuracy  
🔹 **Deploying the model as a real-time API** for live Question Answering  
🔹 **Comparing GPT-Neo’s performance with other LLMs** like GPT-J, BLOOM, Falcon  

---

## **👨‍💻 Project Contributors**  

- **[Bhimavarapu Saiteja Reddy](https://www.linkedin.com/in/Teja220)** – Model fine-tuning, optimization, and training.  

📧 **For any questions or collaborations, feel free to reach out:**  
- **LinkedIn:** https://www.linkedin.com/in/Teja220  
- **Email:** saitejaredyy@gmail.com  
