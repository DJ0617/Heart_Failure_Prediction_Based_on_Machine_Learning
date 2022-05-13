# EN.530.641 Final Project ----- Task 1: Machine Learning: Heart Failure Prediction
# Team 9: Daijie Bao & Han Gao
# Algorithms of this project task was created by Daijie Bao
# import Necessary Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
# import datasets for task 1
task1_datasets = pd.read_csv('heart.csv')
print(task1_datasets.info())
# preprocessing datasets for task 1 --- Created by Daijie Bao
for cols in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
    lb_encoder = LabelEncoder()
    task1_datasets[cols] = lb_encoder.fit_transform(task1_datasets[cols])
X = task1_datasets.drop(['HeartDisease'], axis=1)
Y = task1_datasets['HeartDisease']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
n_cols = len(X_train.columns)  # input shape for artificial neural network
Scaler = StandardScaler()
scaled_X_train = Scaler.fit_transform(X_train)
scaled_X_test = Scaler.fit_transform(X_test)
# build classification models for task 1
# Model 1: support vector machine model for task 1
grid_parameters_svc = {'C': [0.1, 1, 10, 100, 1000],
                       'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                       'kernel': ['rbf', 'poly', 'sigmoid']}
svc_model_gs = GridSearchCV(SVC(), grid_parameters_svc, refit=True, verbose=3)
svc_model_gs.fit(scaled_X_train, Y_train)
svc_model_gs_prediction = svc_model_gs.predict(scaled_X_train)
print('The best parameters of the support vector machine are: ', svc_model_gs.best_params_)
svc_accuracy = svc_model_gs.best_score_
print('The model accuracy of the support vector machine is: ', svc_accuracy)
# Model 2: Artificial Neural Network model for task 1
ANN_model = keras.Sequential()
ANN_model.add(keras.Input(shape=(n_cols, )))
ANN_model.add(layers.Dense(25, activation='relu'))
ANN_model.add(layers.Dense(15, activation='relu'))
ANN_model.add(layers.Dense(1, activation='sigmoid'))
ANN_model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
ANN_model_history = ANN_model.fit(X_train, Y_train, validation_split=0.9, epochs=160)
Y_prediction = ANN_model.predict(X_test)
ANN_model_accuracy = ANN_model_history.history['accuracy'][-1]
print('The model accuracy of Artificial Neural Network Model is: ', ANN_model_accuracy)
# Model 3: LDA model for task 1
LDA_model = LinearDiscriminantAnalysis()
LDA_model.fit(X_train.values, Y_train)
LDA_prediction = LDA_model.predict(X_test.values)
LDA_model_error_set = np.square(Y_test - LDA_prediction)
LDA_model_error = np.mean(LDA_model_error_set)
LDA_model_accuracy = LDA_model.score(X_test.values, Y_test)
print('The model accuracy of LDA model is: ', LDA_model_accuracy)
# Model Accuracy comparison
model_name = ['SVM_model', 'ANN_model', 'LDA_model']
Accuracy = [svc_accuracy, ANN_model_accuracy, LDA_model_accuracy]
Accuracy_table = pd.DataFrame({'Model': model_name, 'Accuracy': Accuracy})
Accuracy_table.sort_values(by='Accuracy', ascending=True)
sns.barplot(x='Model', y='Accuracy', data=Accuracy_table).set(title='Model Accuracy of Machine Learning Task')
plt.show()




