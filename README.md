# HR-Based Classifier

This project is an HR-based classifier developed during my internship at Jio Platforms Ltd. It leverages Long Short-Term Memory (LSTM) neural networks with an Adam optimizer to classify HR-related queries into specific categories, providing an automated solution for HR management.

## Table of Contents

- [Introduction](#introduction)
- [Project Objectives](#project-objectives)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Overall Working](#overall-working)
- [Use Cases](#use-cases)
- [Installation](#installation)
- [Results](#results)
- [Conclusions](#conclusions)
- [License](#license)

## Introduction

In the modern workplace, HR departments face the challenge of managing a high volume of inquiries from employees regarding various topics. This project aims to automate the classification of these queries, improving response time and allowing HR professionals to focus on more strategic tasks. 

## Project Objectives

- Develop an effective classifier that categorizes HR-related queries into predefined categories.
- Utilize an LSTM neural network to capture the sequential nature of the text data.
- Implement the Adam optimizer for efficient training of the model.

## Technologies Used

- Python
- TensorFlow/Keras
- NumPy
- Pandas
- Scikit-learn
- JSON for data storage

## Dataset

The dataset consists of various HR-related queries along with their corresponding categories, such as Leaves, Payroll, Healthcare, Recruitment, etc. The data was preprocessed to ensure it was clean and properly formatted for training.

## Model Architecture

The LSTM model architecture includes:

- **Embedding Layer**: Converts integer-encoded words into dense vectors of fixed size, allowing the model to learn word representations.
- **LSTM Layers**: Capture temporal dependencies in the sequences of text data, which is crucial for understanding the context of queries.
- **Dense Layers**: Process the outputs from the LSTM layer, providing a softmax output for multi-class classification.

## Overall Working

1. **Data Preprocessing**: 
   - The text data is tokenized and converted into sequences of integers. Each word in the dataset is mapped to an integer index based on its frequency.
   - The sequences are padded to ensure uniform input size for the LSTM model.

2. **Model Training**:
   - The model is built and compiled, incorporating layers for embedding, LSTM processing, and dense output.
   - It is trained on the preprocessed dataset, optimizing its weights to minimize the classification loss.

3. **Interactive Query Classification**:
   - After training, the model can classify new HR-related queries. Users can input queries, and the model predicts the relevant category.
   - Feedback from users can be incorporated to fine-tune the model, allowing for continuous improvement and adaptation to new data.

4. **Model Saving and Fine-tuning**:
   - The trained model and tokenizer configuration are saved for future use, and the model can be fine-tuned based on user interactions, allowing for on-the-fly learning.

## Use Cases

- **Employee Query Handling**: Automate responses to common HR inquiries, reducing response time and improving employee satisfaction.
- **Resource Allocation**: Identify trends in employee queries to allocate HR resources more effectively.
- **Training Needs Assessment**: Classify queries related to training and development, helping HR identify areas for employee growth.
- **Performance Management**: Streamline performance-related inquiries, providing managers with insights into employee concerns.

## Code Explanation

### 1. Imports and Setup

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   https://github.com/Anindya-nub/Hr-chatbot-query-classifier.git
