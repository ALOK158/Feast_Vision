# 📌 FeastAI

## 📝 Project Overview

This project implements **Food101** image classification using **EfficientNet** via **transfer learning**. The goal is to classify food images into 101 categories efficiently. The model is trained on the **Food101 dataset** and fine-tuned for improved accuracy.

---

## 🚀 Features

- ✅ Uses **EfficientNet (B0-B7)** as a feature extractor.
- ✅ Fine-tunes top layers while freezing lower layers.
- ✅ Implements **data augmentation, batch normalization, dropout** for better generalization.
- ✅ Saves trained models in `.h5` or `.pb` format for deployment.

---

## 📂 Repository Structure

```
/food101-efficientnet
  ├── notebooks/          # Jupyter Notebooks for experiments
  ├── README.md           # Project overview & instructions
```

---

## 📦 Installation & Setup

### 1️⃣ Clone the repository:

```sh
git clone https://github.com/yourusername/food101-efficientnet.git
cd food101-efficientnet
```

### 2️⃣ Install dependencies:

```sh
pip install -r requirements.txt
```

### 3️⃣ Download the Food101 dataset:

```python
import tensorflow_datasets as tfds
dataset, info = tfds.load("food101", as_supervised=True, with_info=True)
```

---

## 🏗 Model Training

▶️ Run the training script:

```sh
python src/train.py
```

Or use the Jupyter Notebook in `notebooks/` for step-by-step training.

---

## 📊 Evaluation

📌 After training, evaluate the model:

```sh
python src/evaluate.py
```

---

## 🚀 Deployment

### 🔹 Deploy the model as an API:

```sh
python src/deploy.py
```

### 🔹 Integrate into a web app using **Streamlit**:

```sh
streamlit run src/app.py
```

---

## 🎯 Future Plans

- ✅ Train on **Food101 dataset** with EfficientNet.
- ⏳ Convert model to **TF Lite** or **ONNX** for deployment.
- ⏳ Develop an **API (FastAPI/Flask)** for serving predictions.
- ✅  Build a **web app** for user interaction (Streamlit/Flask).
  

---

## 🤝 Contributing

💡 Feel free to submit **issues, feature requests, or pull requests** to improve the project.

---

## 📜 License

📌 This project is open-source and available under the **MIT License**.


