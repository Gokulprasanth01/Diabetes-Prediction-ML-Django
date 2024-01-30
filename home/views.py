from Cryptodome.Cipher import AES
import mysql.connector
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import joblib
import mysql.connector
from django.shortcuts import render
from django.views.decorators.cache import cache_control
import base64

key = b'Sixteen byte key'
fixed_iv = bytes([0] * 16)

classifier = joblib.load('accounts/model/ran.pkl')
encoder = joblib.load('accounts/model/sac.pkl')


def encrypt(msg):
    cipher = AES.new(key, AES.MODE_EAX, nonce=fixed_iv)
    ciphertext, tag = cipher.encrypt_and_digest(str(msg).encode('utf-8'))

    # Use Base64 encoding for ciphertext and tag
    ciphertext_base64 = base64.b64encode(ciphertext).decode('utf-8')
    tag_base64 = base64.b64encode(tag).decode('utf-8')

    return f"{ciphertext_base64}:{tag_base64}"


def base64_string_to_int(base64_string):
    # Decode Base64 and convert to integer
    return int.from_bytes(base64.b64decode(base64_string), byteorder='big', signed=False)


def index(request):
    return render(request, 'index.html')


def store_results_in_database(gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level,
                              blood_glucose_level):
    try:
        con = mysql.connector.connect(host='localhost', user='root', password='Itachi@001')

        if con.is_connected():
            cursor = con.cursor()
            database_name = 'healthcare1'
            create_database_query = f"CREATE DATABASE IF NOT EXISTS {database_name}"
            cursor.execute(create_database_query)
            cursor.execute(f"USE {database_name}")

            normal_table_name = 'diabetes'
            create_normal_table_query = f"CREATE TABLE IF NOT EXISTS {normal_table_name} (id INT AUTO_INCREMENT PRIMARY KEY, \
                                          gender VARCHAR(255), age VARCHAR(255), hypertension VARCHAR(255), \
                                          heart_disease VARCHAR(255), smoking_history VARCHAR(255), \
                                          bmi VARCHAR(255), HbA1c_level VARCHAR(255), blood_glucose_level VARCHAR(255))"
            cursor.execute(create_normal_table_query)

            insert_normal_query = f"INSERT INTO {normal_table_name} (gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
            normal_data = (
            gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level)
            cursor.execute(insert_normal_query, tuple(map(str, normal_data)))

            con.commit()
            cursor.close()
            con.close()

    except Exception as e:
        print(f"Error storing results in the database: {e}")


@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def index(request):
    return render(request, 'index.html')


@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def diabetes(request):
    value = ''
    if request.method == "POST":
        gender = str(request.POST['gender'])
        age = float(request.POST['age'])
        hypertension = float(request.POST['hypertension'])
        heart_disease = float   (request.POST['heart_disease'])
        smoking_history = str(request.POST['smoking_history'])
        bmi = float(request.POST['bmi'])
        HbA1c_level = float(request.POST['HbA1c_level'])
        blood_glucose_level = float(request.POST['blood_glucose_level'])

        en_gen = base64_string_to_int(encrypt(gender).split(':')[0])
        en_age = base64_string_to_int(encrypt(age).split(':')[0])
        en_hy = base64_string_to_int(encrypt(hypertension).split(':')[0])
        en_hd = base64_string_to_int(encrypt(heart_disease).split(':')[0])
        en_bmi = base64_string_to_int(encrypt(bmi).split(':')[0])
        en_sh = base64_string_to_int(encrypt(smoking_history).split(':')[0])
        en_hl = base64_string_to_int(encrypt(HbA1c_level).split(':')[0])
        en_bl = base64_string_to_int(encrypt(blood_glucose_level).split(':')[0])

        new_instance = [[en_gen, en_age, en_hy, en_hd, en_bmi, en_sh, en_hl, en_bl]]
        # Check the number of features before applying the encoder
        new_instance_encoded = encoder.transform(new_instance)
        encrypted_prediction = classifier.predict(new_instance_encoded)

        if encrypted_prediction[0] == 1:
            value = 'Positive'
        elif encrypted_prediction[0] == 0:
            value = 'Negative'
            store_results_in_database(gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level)

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

