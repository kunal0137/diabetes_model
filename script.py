#Mount ( "ae7dcd76-c8b4-49ca-80ca-15a88e4f8051" , "kunal" ) ;
#SetContext(context=["ae7dcd76-c8b4-49ca-80ca-15a88e4f8051"]);


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

#diabetesDF=Diabetes_KP_FRAME163341.drop(['DIABETES_ML_UNIQUE_ROW_ID','id','Drug','frame'], axis=1)
diabetesDF=Diabetes_KP_FRAME163341.drop(['DIABETES_ML_UNIQUE_ROW_ID','id','Drug','frame', 'MANUFACTURER_MAKING_PAYMENT_NAME','DRUG'], axis=1)

diabetesDF = diabetesDF.sample(frac=1).reset_index(drop=True)





## split the data frame##
dfTrain = diabetesDF[:650]
dfTest = diabetesDF[650:750]
dfCheck = diabetesDF[750:]


## splitting up the training data - split the label and features - shifting to numpy
trainLabel = np.asarray(dfTrain['Outcome'])
trainData = np.asarray(dfTrain.drop('Outcome',1))
testLabel = np.asarray(dfTest['Outcome'])
testData = np.asarray(dfTest.drop('Outcome',1))


## normalizing the data such that the mean is 0 and the stnd dev is 1
means = np.mean(trainData, axis=0)
stds = np.std(trainData, axis=0)
trainData = (trainData - means)/stds
testData = (testData - means)/stds
# np.mean(trainData, axis=0) => check that new means equal 0
# np.std(trainData, axis=0) => check that new stds equal 1



## Training the classification model - load logistic regression
## create an instance of the model here: diabetesCheck
## use the fit function to train the model
diabetesCheck = LogisticRegression()
diabetesCheck.fit(trainData, trainLabel)


## we can use the test data to check the accuracy
accuracy = diabetesCheck.score(testData, testLabel)

print("accuracy = ", accuracy * 100, "%")


coeff = list(diabetesCheck.coef_[0])
labels = list(dfTrain.drop('Outcome',1))
features = pd.DataFrame()
features['Features'] = labels
features['importance'] = coeff
features.sort_values(by=['importance'], ascending=True, inplace=True)
features['positive'] = features['importance'] > 0

path=kunal+'/diabetes_model/diabetesModel.pkl'
dataPath=kunal+'/diabetes_model/dftrain.csv'
dfTrain.to_csv(dataPath);
## save the model
joblib.dump([diabetesCheck, means, stds], path)

## just testing that the model saved right
diabetesLoadedModel, means, stds = joblib.load(path)
accuracyModel = diabetesLoadedModel.score(testData, testLabel)


## do a manual check
sampleData = dfCheck[:1]

# prepare sample
sampleDataFeatures = np.asarray(sampleData.drop('Outcome',1))
sampleDataFeatures = (sampleDataFeatures - means)/stds
# predict
predictionProbability = diabetesLoadedModel.predict_proba(sampleDataFeatures)
prediction = diabetesLoadedModel.predict(sampleDataFeatures)
print('Probability:', predictionProbability)
print('prediction:', prediction)



predictionProbability = diabetesCheck.predict_proba(sampleDataFeatures)
prediction = diabetesCheck.predict(sampleDataFeatures)
print('Probability:', predictionProbability)
print('prediction:', prediction)
diabetesCheck
