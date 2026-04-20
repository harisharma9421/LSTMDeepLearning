# LSTM-Based Sequence Prediction System with FastAPI Deployment

## Project Overview
This project implements a **Sequence Prediction System using Long Short-Term Memory (LSTM)** networks.  
The model is trained to **predict the next word in a sequence** using a text dataset.

The trained deep learning model is deployed using **FastAPI**, allowing real-time predictions via API endpoints.

This project demonstrates a complete AI pipeline including:

- Dataset preprocessing
- Sequence generation
- LSTM model training
- Model evaluation
- API deployment

---

# Objectives

- Build a deep learning model using LSTM
- Predict the next word in a sentence
- Deploy the model using FastAPI
- Demonstrate real-time AI inference through API endpoints

---

# Dataset Used

Dataset: **Tiny Shakespeare Dataset**

Source:
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

This dataset contains dialogues from Shakespeare's plays and is widely used for sequence prediction tasks.

Example dataset text:


First Citizen:
Before we proceed any further, hear me speak.


---

# Data Preprocessing

The dataset was preprocessed using the following steps:

### 1. Text Cleaning
- Converted text to lowercase
- Removed punctuation and special characters

Example:


My Lord, I will not hear you speak.


After cleaning:


my lord i will not hear you speak


---

### 2. Tokenization

Each word is converted into a numerical index using a tokenizer.

Example:


my lord i will not hear you speak


Tokenized sequence:


[34, 120, 5, 87, 22, 56, 13, 90]


---

### 3. Sequence Generation

Training sequences are generated as input-output pairs.

Example sentence:


my lord i will not hear


Generated sequences:


my lord → i
my lord i → will
my lord i will → not
my lord i will not → hear


---

### 4. Padding

Since sequence lengths vary, padding is applied to make them uniform.

Example padded sequence:


[0, 0, 34, 120, 5, 87]


---

# Model Architecture

The model architecture consists of the following layers:


Embedding Layer
↓
LSTM Layer
↓
Dropout Layer
↓
LSTM Layer
↓
Dense Output Layer


### Layer Description

**Embedding Layer**
- Converts word indices into dense vectors

**LSTM Layer**
- Learns sequential dependencies

**Dropout Layer**
- Prevents overfitting

**Dense Layer**
- Predicts probability of next word

---

# LSTM Mathematical Model

LSTM uses gates to control information flow.

### Forget Gate


ft = σ(Wf[ht−1, xt] + bf)


### Input Gate


it = σ(Wi[ht−1, xt] + bi)


### Candidate Memory


C̃t = tanh(Wc[ht−1, xt] + bc)


### Cell State Update


Ct = ft * Ct−1 + it * C̃t


### Output Gate


ot = σ(Wo[ht−1, xt] + bo)
ht = ot * tanh(Ct)


---

# Model Training

Training configuration:

| Parameter | Value |
|--------|--------|
Loss Function | Sparse Categorical Crossentropy |
Optimizer | Adam |
Batch Size | 64 |
Epochs | 40 |

---

# Model Performance

Training Accuracy:

**~0.40 – 0.55**

Accuracy increased steadily while loss decreased during training.

---

# Training Accuracy Graph

![Accuracy Graph](screenshots/Accuracy.png)

---

# Training Loss Graph

![Loss Graph](screenshots/Loss.png)

---

# Prediction Examples

Input:


my lord


Output:


my lord i will not hear me speak


---

# Example Prediction Output

![Prediction Example](screenshots/prediction_output.png)

---

# API Deployment using FastAPI

The trained model was deployed using **FastAPI**.

### Run API


uvicorn app:app --reload


Server runs at:


http://127.0.0.1:8000


Swagger UI:


http://127.0.0.1:8000/docs


---

# API Endpoint Example

### Predict Next Word


/predict


Example request:


text = "my lord"


Example response:


{
"input_text": "my lord",
"next_word": "i"
}


---

# FastAPI Swagger Interface

![Swagger API](screenshots/api_swagger.png)

---

# System Architecture


Dataset
↓
Data Preprocessing
↓
Tokenization
↓
Sequence Generation
↓
LSTM Model Training
↓
Prediction
↓
FastAPI Deployment
↓
Real-Time API Response


---

# Applications

Sequence prediction systems are used in:

- Chatbots
- Text autocomplete
- AI assistants
- Machine translation
- Text generation

---

# Conclusion

This project successfully implements an LSTM-based sequence prediction system capable of predicting the next word in a sequence.

The trained model was deployed using FastAPI to enable real-time predictions through API endpoints. This project demonstrates the integration of deep learning and web deployment to build an industry-relevant AI system.

---

# AI Tool Acknowledgement

Tool Used: **ChatGPT**

Purpose:
- Understanding LSTM architecture
- Debugging model training
- Assistance in FastAPI deployment
- Documentation preparation
