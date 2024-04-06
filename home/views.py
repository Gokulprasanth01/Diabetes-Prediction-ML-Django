# Import necessary libraries
from Cryptodome.Cipher import AES
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64
import joblib
import mysql.connector
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from django.shortcuts import render
from django.views.decorators.cache import cache_control


# Function to pad data for AES encryption
def pad_data(data):
    block_size = algorithms.AES.block_size // 8
    padding_length = block_size - (len(data) % block_size)
    padding = bytes([padding_length] * padding_length)
    return data + padding


# Function to encrypt data using AES
def encrypt_data(key, data):
    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
    encryptor = cipher.encryptor()
    return encryptor.update(pad_data(data)) + encryptor.finalize()


# Function to encrypt each cell in a DataFrame
def encrypt_dataframe(df, key):
    return df.applymap(lambda cell: encrypt_data(key, str(cell).encode('utf-8')))


# Function to convert bytes to integer
def base64_to_int(df):
    return int.from_bytes(df, byteorder='big')  # or 'little' depending on your byte order


# Load the pre-trained model and encoder
classifier = joblib.load('accounts/model/ran.pkl')
encoder = joblib.load('accounts/model/sac.pkl')

# Define a fixed key for encryption (replace with secure key management)
key = b'Sixteen byte key'


# Function to store results in the database


# Define the Django views

@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def index(request):
    return render(request, 'index.html')


@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def diabetes(request):
    value = ''
    if request.method == "POST":
        # Extract user input from the form
        gender = str(request.POST['gender'])
        age = float(request.POST['age'])
        hypertension = float(request.POST['hypertension'])
        heart_disease = float(request.POST['heart_disease'])
        smoking_history = str(request.POST['smoking_history'])
        bmi = float(request.POST['bmi'])
        HbA1c_level = float(request.POST['HbA1c_level'])
        blood_glucose_level = float(request.POST['blood_glucose_level'])

        # Encrypt and convert user input to integers
        en_gen = base64_to_int(encrypt_data(key, gender.encode('utf-8')))
        en_age = base64_to_int(encrypt_data(key, str(age).encode('utf-8')))
        en_hy = base64_to_int(encrypt_data(key, str(hypertension).encode('utf-8')))
        en_hd = base64_to_int(encrypt_data(key, str(heart_disease).encode('utf-8')))
        en_bmi = base64_to_int(encrypt_data(key, str(bmi).encode('utf-8')))
        en_sh = base64_to_int(encrypt_data(key, smoking_history.encode('utf-8')))
        en_hl = base64_to_int(encrypt_data(key, str(HbA1c_level).encode('utf-8')))
        en_bl = base64_to_int(encrypt_data(key, str(blood_glucose_level).encode('utf-8')))

        # Create a new instance for prediction
        new_instance = [[en_gen, en_age, en_hy, en_hd, en_bmi, en_sh, en_hl, en_bl]]

        # Encode features using the pre-trained encoder
        new_instance_encoded = encoder.transform(new_instance)
        print(new_instance_encoded)
        # Make predictions using the pre-trained Random Forest Classifier
        encrypted_prediction = classifier.predict(new_instance_encoded)

        # Determine the result based on the prediction
        if encrypted_prediction[0] == 1:
            value = 'Positive'
        elif encrypted_prediction[0] == 0:
            value = 'Negative'
            # Store results in the database for negative predictions
            # store_results_in_database(gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level,
            #                           blood_glucose_level)

    return render(request, 'diabetes.html', {'context': value})


def breast(request):
    # Reading training data set.

    df = pd.read_csv('data/Breast_train.csv')
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1]
    print(X.shape, Y.shape)

    # Reading data from user.

    value = ''
    if request.method == 'POST':

        radius = float(request.POST['radius'])
        texture = float(request.POST['texture'])
        perimeter = float(request.POST['perimeter'])
        area = float(request.POST['area'])
        smoothness = float(request.POST['smoothness'])

        # Creating our training model.

        rf = RandomForestClassifier(
            n_estimators=16, criterion='entropy', max_depth=5)
        rf.fit(np.nan_to_num(X), Y)

        user_data = np.array(
            (radius,
             texture,
             perimeter,
             area,
             smoothness)
        ).reshape(1, 5)

        predictions = rf.predict(user_data)
        print(predictions)

        if int(predictions[0]) == 1:
            value = 'have'
        elif int(predictions[0]) == 0:
            value = "don\'t have"

    return render(request,
                  'breast.html',
                  {
                      'context': value,

                  })


def heart(request):
    df = pd.read_csv('data/Heart_train.csv')
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1:]

    value = ''

    if request.method == 'POST':

        blood_glucose_level = float(request.POST['blood_glucose_level'])
        sex = float(request.POST['sex'])
        cp = float(request.POST['cp'])
        trestbps = float(request.POST['trestbps'])
        chol = float(request.POST['chol'])
        fbs = float(request.POST['fbs'])
        restecg = float(request.POST['restecg'])
        thalach = float(request.POST['thalach'])
        exang = float(request.POST['exang'])
        oldpeak = float(request.POST['oldpeak'])
        slope = float(request.POST['slope'])
        ca = float(request.POST['ca'])
        thal = float(request.POST['thal'])

        user_data = np.array(
            (blood_glucose_level,
             sex,
             cp,
             trestbps,
             chol,
             fbs,
             restecg,
             thalach,
             exang,
             oldpeak,
             slope,
             ca,
             thal)
        ).reshape(1, 13)

        rf = RandomForestClassifier(
            n_estimators=16,
            criterion='entropy',
            max_depth=9
        )

        rf.fit(np.nan_to_num(X), Y)
        rf.score(np.nan_to_num(X), Y)
        predictions = rf.predict(user_data)

        if int(predictions[0]) == 1:
            value = 'have'
        elif int(predictions[0]) == 0:
            value = "don\'t have"

    return render(request,
                  'heart.html',
                  {
                      'context': value,

                  })

