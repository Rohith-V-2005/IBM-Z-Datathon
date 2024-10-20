import pickle
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def load_wesad_data(subject_path):
    with open(subject_path+'\S2.pkl', 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    bvp = data['signal']['wrist']['BVP'][:, 0] 
    eda = data['signal']['wrist']['EDA'][:, 0] 
    temp = data['signal']['wrist']['TEMP'][:, 0] 
    acc = data['signal']['wrist']['ACC']  
    acc_magnitude = np.sqrt(np.sum(acc**2, axis=1))
    labels = data['label']
    return bvp, eda, temp, acc_magnitude, labels

def preprocess_wesad_signals(bvp, eda, temp, acc_magnitude, labels, seq_length=10):
    X, y = [], []
    min_length = min(len(bvp), len(eda), len(temp), len(acc_magnitude))
    bvp = bvp[:min_length]
    eda = eda[:min_length]
    temp = temp[:min_length]
    acc_magnitude = acc_magnitude[:min_length]
    combined_signals = np.stack((bvp, eda, temp, acc_magnitude), axis=1)
    for i in range(len(combined_signals) - seq_length):
        X.append(combined_signals[i:i+seq_length]) 
        y.append(labels[i+seq_length]) 
    X = np.array(X)
    y = np.array(y)
    valid_indices = np.where((y == 1) | (y == 2) | (y == 3) | (y == 4))
    X = X[valid_indices]
    y = y[valid_indices]
    
    return X, y

def encode_labels(y):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder

def load_dataset_wesad(subject_path, seq_length=10, test_size=0.2):
    bvp, eda, temp, acc_magnitude, labels = load_wesad_data(subject_path)
    X, y = preprocess_wesad_signals(bvp, eda, temp, acc_magnitude, labels, seq_length)
    y_encoded, encoder = encode_labels(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test, encoder

# Step 4: Build the LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))  

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
if __name__ == '__main__':
    data_path = r'C:\Users\ROHITH\Documents\ibm\WESAD\S2' 
    X_train, X_test, y_train, y_test, encoder = load_dataset_wesad(data_path)
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))  
    model.summary()  
    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=1)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f'Test Accuracy: {accuracy:.2f}')
    model.save('wesad_emotion_model.h5')
