import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import f1_score, classification_report
import pdb
from sklearn.svm import LinearSVC
import re

# citire date
# fiecare rand din trainingData/validationData/testData are un id si continut
# fiecare rand din trainingLabels/validationLabels are un id si eticheta 0/1 
with open('train_samples.txt', mode="r", encoding="utf-8") as trainSample:
  trainingData = trainSample.readlines()
  print(type(trainingData))
with open('train_labels.txt', mode="r", encoding="utf-8") as trainLabels:
  trainingLabels = trainLabels.readlines()

with open('validation_samples.txt', mode="r", encoding="utf-8") as validationSample:
  validationData = validationSample.readlines()
with open('validation_labels.txt', mode="r", encoding="utf-8") as valLabels:
  validationLabels = valLabels.readlines()

with open('test_samples.txt', mode="r", encoding="utf-8") as testSample:
  testData = testSample.readlines()
print("Am trecut de incarcarea datelor...")

# Crearea unei liste cu toate cuvintele din datele de antrenare
allWords = []
for line in trainingData:
  words = line.split()
  for i, w in enumerate(words):
    if i != 0 and w not in allWords:
      allWords.append(w)
# Transformarea lui all in array
allWords = np.array(allWords)
# Nr de cuvinte distincte din date
noOfWords = len(allWords)
# print("noOfWords:")
# print(noOfWords)

def getFeatures(data):
   features = np.zeros((len(data), noOfWords))
   for i, document in enumerate(data):
     for word in document:
       if word in allWords:
         features[i, np.where(allWords == word)[0][0]] += 1
   return features

# Frecventa cuvintelor in fiecare linie din fiecare fisier 
trainFeatures = getFeatures(trainingData)
validationFeatures = getFeatures(validationData)
testFeatures = getFeatures(testData) 

# print("Features shape: (train, validation, test)")
# print(trainFeatures.shape)
# print(validationFeatures.shape)
# print(testFeatures.shape)

# normalizare date
scaledTrainFeatures = ()
scaledValidationFeatures = ()
scaledTestFeatures = ()
scaler = None

scaler = preprocessing.Normalizer(norm = 'l2')

if scaler is not None:
  scaler.fit(trainFeatures)
  scaledTrainFeatures = scaler.transform(trainFeatures)
  scaledValidationFeatures = scaler.transform(validationFeatures) 
  scaledTestFeatures = scaler.transform(testFeatures) 
else:
  scaledTrainFeatures = trainFeatures
  scaledValidationFeatures = validationFeatures
  scaledTestFeatures = testFeatures

# Creare svm
svmModel = svm.SVC(C = 100, kernel = 'linear')
# svmModel = svm.SVC(C = 3, kernel = 'linear')
# svmModel = svm.LiniarSVC()

# Antrenarea pe trainingData
svmModel.fit(scaledTrainFeatures, trainingLabels)

# Aplicarea modelului pe datele de validare
predictedLabelsValidation = svmModel.predict(scaledValidationFeatures) 

# Verificare pe datele de validare
tp = 0 # Nr de cazuri din clasa 1 pe care modelul le-a clasificat ca fiind din clasa 1
tn = 0 # Nr de cazuri din clasa 0 pe care modelul le-a clasificat ca fiind din clasa 0
fp = 0 # Nr de cazuri din clasa 1 pe care modelul le-a clasificat ca fiind din clasa 0
fn = 0 # Nr de cazuri din clasa 0 pe care modelul le-a clasificat ca fiind din clasa 1

validationTrueLabels = [] # lista cu etichetele date in datele de validare
for line in validationLabels:
  validationTrueLabels.append(line.split()[1])

validationPredLabels = [] # lista cu predictiile pentru datele de validare
for line in predictedLabelsValidation:
  validationPredLabels.append(line.split()[1])

for i, f in enumerate(validationPredLabels):
  if validationPredLabels[i] == '1':
    if validationTrueLabels[i] == '1':
      tp += 1
    else:
      fp += 1
  if validationPredLabels[i] == '0':
    if validationTrueLabels[i] == '0':
      tn += 1
    else:
      fn += 1
# 1. confusion matrix
print(tn, fp)
print(fn, tp)
# 2. f1 score
precision = tp/(tp+fp)
recall = tp/(tp+fn)

f1 = 2* (precision * recall)/(precizie + recall)
print(f1) # 0.5175242356450409

# Aplicarea modelului pe datele de testare
predictedLabelsSvm = svmModel.predict(scaledTestFeatures)
# print("predictedLabelsSvm")
# print(predictedLabelsSvm)

# Incarcare rezultate in fisier
testLab = [] # lista cu id-urile propozitiilor din din datele de test
for line in testData:
  testLab.append(line.split()[0])
# print("testLab")
# print(testLab)

pred = [] # lista cu predictiile pentru datele de test
for i in predictedLabelsSvm:
  s = re.split(r'\t+', i.rstrip('\t'))
  pred.append(s[1])
# print("pred")
# print(pred)

submission = pd.DataFrame({'id': testLab,'label': pred})
name = 'submission.csv'
submission.to_csv(name, index = False)
