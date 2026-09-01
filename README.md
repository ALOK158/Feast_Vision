# 📌 FeastAI

>Now the project is getting upgraded and moving beyond multi-class EfficientNet classification toward VLM-powered food understanding. 

## Project Overview

This project implements **Food101** image classification using **EfficientNet** via **transfer learning**. The goal is to classify food images into 101 categories efficiently. The model is trained on the **Food101 dataset** and fine-tuned for improved accuracy.

## Features

- ✅ Uses **EfficientNetB0/B1** as a feature extractor.
- ✅ Fine-tunes top layers while freezing lower layers.
- ✅ Implements **data augmentation, batch normalization, dropout** for better generalization.
- ✅ Mixed precision training for faster convergence.
- ✅ Exports to **ONNX** for lightweight framework for easy deployment.


##  Repository Structure

```
Feast_Vision/
├── FeastAI.ipynb # Training notebook — data pipeline, EfficientNet model, training loop
├── stream.py # Streamlit app for inference
├── convert_2_onnx.py # Converts trained Keras model to ONNX
├── food_classifier_model/
│   └── food_classifier.onnx # Deployment-ready ONNX model
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

> Note: `food_classifier.keras` (the original trained model, ~131MB) is kept locally and is not pushed to this repo  only the converted `.onnx` model is published.


##  Installation & Setup

###  Clone the repository:

```sh
git clone https://github.com/ALOK158/Feast_Vision.git
cd Feast_Vision
```

### 2️⃣ Install dependencies:

```sh
pip install -r requirements.txt
```

### 3️⃣ Download the Food101 dataset (handled inside the notebook):

```python
import tensorflow_datasets as tfds
dataset, info = tfds.load("food101", as_supervised=True, with_info=True)
```

---

## 🏗 Model Training

Training is done inside **`FeastAI.ipynb`**  open it in Jupyter/Colab and run through the cells (data loading, preprocessing, EfficientNet transfer learning, training with early stopping + LR reduction callbacks).

---

## 🔄 Model Conversion (ONNX)

Convert the trained Keras model to ONNX for lighter, framework-agnostic deployment:

```sh
python convert_2_onnx.py --model food_classifier_model/food_classifier.keras --output food_classifier_model/food_classifier.onnx
```

This reduces the model from ~131MB (Keras) to ~13MB (ONNX) with the same accuracy  and enabling deployment to web and mobile app without needing the full TensorFlow runtime and huge storage.


##  Deployment

🔗 **Live app:** [https://feastvision-gwapvesrky9vmhazuff2oc.streamlit.app/](https://feastvision-gwapvesrky9vmhazuff2oc.streamlit.app/)

Run the Streamlit app locally:

```sh
streamlit run stream.py
```

##  To-Do / Future Plans

- ✅ Train on **Food101 dataset** with EfficientNet Fine tuning.
- ✅  Deployed model on Streamlit Cloud and anyone can use it using our web app
- ✅ Convert model to **ONNX** for lightweight deployment.
- ✅ Build a **web app** for user interaction (Streamlit).
- ⏳ Upgrade to a **VLM (Vision-Language Model)** for open-vocabulary food identification and upgrading from multi class classification of 101 food types.
- ⏳ Explore **Fine_tuning** for better working on our partcualr food identification and info domain.
- ⏳ Will add To-Do list as new ideas comes in.

##  License

📌 This project is open-source and available under the **MIT License**.