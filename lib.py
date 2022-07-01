import numpy as np

np.random.seed(131)

import os
import pandas as pd
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler 
from pandas.api.types import is_string_dtype
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import RepeatedStratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import layers, Input
from tensorflow.keras import optimizers
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import metrics
from keras.layers import Lambda
from keras import backend as K

global resultFolder

def readExcel(filename):
    df = pd.read_excel(filename)
    return df

def readCSV(fileName):
    df = pd.read_csv(fileName)
    return df

def checkRequiredColumns(poultryFileName,independantVariables,targetVariable):
   df = readCSV(poultryFileName)
   cols = df.columns
   if targetVariable not in cols:
        print(targetVariable, " not found in dataset file.")
        exit(0)
   for var in independantVariables:
      if var not in cols:
         print(var, " not found in dataset file.")
         exit(0)


def getCategoryVariableNames(df, independantVariables):
    categoryVars = []
    numericVars = []
    binaryVars = []
    for inputVariable in independantVariables:
        if is_string_dtype(df[inputVariable]): 
            values = [item.lower() for item in list(set(list(df[inputVariable]))) if pd.notnull(item)]
            bFlag = True
            for val in values:
                 if not (val == "yes" or val == "y" or val == "no" or val == "n" or val == "na"):
                    bFlag = False
            if bFlag:
               binaryVars.append(inputVariable)
            else:
               categoryVars.append(inputVariable)
        else:
            numericVars.append(inputVariable)    
    return binaryVars, categoryVars, numericVars

def prepareData(poultryFileName, independantVariables, targetVariable, sampleTypes, verbose=False):
#    df = readExcel(poultryFileName)    
    df = readCSV(poultryFileName)
    #Get rows where target value is not NaN
    dfNotNaN = df[df[targetVariable].notna().any(1)]

    for feature in independantVariables:
        dfNotNaN = dfNotNaN[dfNotNaN[feature].notna()]

    #Remove SampleType from independant variable list
    if 'SampleType' in independantVariables:
        independantVariables.remove('SampleType')
    binaryVars, categoryVars, numericVars = getCategoryVariableNames(df, independantVariables)

#    print("CategoryVariables -> ", categoryVars)
#    print("\nNumericVariables -> ", numericVars)
#    print("\nBinaryVariables -> ", binaryVars)
    return dfNotNaN, binaryVars, categoryVars, numericVars

def labelEncode(y):
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(y)

def getTargetVales(dfVar, targetVariable):
    
    y  = labelEncode(np.asarray(dfVar[targetVariable[0]].values.tolist()))

    for i in range(1, len(targetVariable)):
        y |= labelEncode(np.asarray(dfVar[targetVariable[i]].values.tolist()))

    #label encode target values
    return y

def getOneHotFit(x1):
    onehot_encoder = OneHotEncoder(sparse=False)
    return onehot_encoder.fit(x1)

def getOneHotTransform(onehot_encoder, x1):
    return onehot_encoder.transform(x1)

def labelEncodeFit(y):
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    return label_encoder

def labelEncodeTransform(enc, y):
    return enc.transform(y)

def getBinaryValues(dfVar, binaryVariables):
    enc = labelEncodeFit(['No','Yes'])
    y = []
    if len(binaryVariables) > 0:
       dfNew = dfVar[binaryVariables]
       dfNew = dfNew.replace('^(?i)y.*$', 'Yes', regex=True)
       dfNew = dfNew.replace('^(?i)n.*$', 'No', regex=True)
       y = labelEncodeTransform(enc, np.asarray(dfNew[binaryVariables[0]].values.tolist()))
       for i in range(1, len(binaryVariables)):
           y = np.vstack((y,labelEncodeTransform(enc, np.asarray(dfNew[binaryVariables[i]].values.tolist()))))

    #label encode binary variables
    return y.T

def getXTranform(dfAll, dfPrepare, binaryVars, categoryVars, numericVars):

    #Separate category variable input values
    xEnc = np.asarray(dfAll[categoryVars].values.tolist())
    #prepare one hot encoder
    onc = getOneHotFit(xEnc)

    #Separate category variable input values and numerical values
    x1 = np.asarray(dfPrepare[categoryVars].values.tolist())
    x2 = np.asarray(dfPrepare[numericVars].values.tolist())
    
    # Concatenate the onehot vector and the numberical vector
    xTransform = getOneHotTransform(onc, x1)
    xWithLabels = pd.DataFrame (xTransform)
    xWithLabels.columns = onc.get_feature_names_out(categoryVars)
    x2WithLabels = pd.DataFrame (x2)
    x2WithLabels.columns = numericVars
    xWithLabels = xWithLabels.combine_first(x2WithLabels)
    X  = np.hstack((xTransform, x2))

    if len(binaryVars) > 0:
       x3 = getBinaryValues(dfPrepare, binaryVars)
       x3WithLabels = pd.DataFrame (x3)
       x3WithLabels.columns = binaryVars

       xWithLabels = xWithLabels.combine_first(x3WithLabels)

       X  = np.hstack((X, x3))
#    print("One hot encoding for category variables...")

#    print("Label encoding for target variable...")
    return xWithLabels, X


def prepareXY(dfNotNaN, sType, binaryVars, categoryVars, numericVars, targetVariable):
    
    dfVar = dfNotNaN
    if sType is not None:
        if sType == 'pre-harvest':
            dfVar = dfNotNaN[(dfNotNaN['SampleType'] == 'Feces') | (dfNotNaN['SampleType'] == 'Soil')]
        else:        
            dfVar = dfNotNaN[dfNotNaN['SampleType'] == sType]

    #Get target values
    Y = getTargetVales(dfVar, targetVariable)
    xWithLabels, X = getXTranform(dfNotNaN, dfVar, binaryVars, categoryVars, numericVars)
    return xWithLabels, Y


def scaler(X, scaleType = None):

    if scaleType == "quantileTranform":
        qt = QuantileTransformer(n_quantiles=10, random_state=0, output_distribution='normal')
        X =  qt.fit_transform(X)   
    elif scaleType == "unitnorm":
        transformer = Normalizer().fit(X)
        X = transformer.transform(X)
    elif scaleType == "robustScale":
        transformer = RobustScaler().fit(X)
        X = transformer.transform(X)
        
    return X 

def predictionModel(X, Y, hLayers,epochs=100, activation='relu', kernel_initializer=None, optimizer='adam'):  
    
    inputDim = X.shape[1]
    # create model
    model = Sequential()    
    firstLayer = True

    for hSize in hLayers:
        if firstLayer:
            firstLayer = False
            if kernel_initializer:
                model.add(Dense(hSize, input_dim=inputDim, activation=activation, kernel_initializer=kernel_initializer))
            else:
                model.add(Dense(hSize, input_dim=inputDim, activation=activation))
        else:
            if kernel_initializer:
                model.add(Dense(hSize, activation=activation, kernel_initializer=kernel_initializer))
            else:
                model.add(Dense(hSize, activation=activation))

    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    loss = model.fit(X, Y, epochs=epochs, batch_size=8, validation_data=(X, Y), shuffle=False,validation_steps=1, verbose=0) 

    return model, loss

def getROC(yTrue, yPred, printPdf, algDetails = "Default"):

        yPred = yPred.reshape(-1)
        yTrue = yTrue.reshape(-1)
#        print("Labels-" + algDetails + "," + ",".join([str(val) for val in yTrue]))
#        print("Predictions, " + ",".join([str(val) for val in yPred]))
        fpr, tpr, thresholds = metrics.roc_curve(yTrue, yPred)
        roc_auc = metrics.auc(fpr, tpr)
     
        plt.ioff()

#        fig = plt.figure(num=None, figsize=(14, 8))
        fig = plt.figure()
        lw = 4
        plt.tight_layout()
        fig.tight_layout()
        fSize = 20
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = fSize
        plt.rcParams['axes.labelsize'] = fSize
        plt.plot(fpr, tpr, color='darkorange',
           lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=fSize)
        plt.ylabel('True Positive Rate', fontsize=fSize)
        plt.legend(loc="lower right")
        plt.xticks(fontsize=fSize)
        plt.yticks(fontsize=fSize)

        axes = plt.gca()
        axes.xaxis.label.set_size(fSize)
        axes.yaxis.label.set_size(fSize)
        plt.savefig(printPdf, format='pdf', bbox_inches='tight',pad_inches=0.5, transparent=True)
        plt.close(fig)

def getMetrics(y_test, yPred):
    
    yPred1 = yPred.reshape(-1)
    y_test = y_test.reshape(-1)   
    yPred = [0 if pr < 0.5 else 1 for pr in yPred1]
    auc1 = metrics.roc_auc_score(y_test, yPred)
    prec1 = metrics.precision_score(y_test, yPred, average="macro")
    rec1 = metrics.recall_score(y_test, yPred, average="macro")
    acc1 = metrics.accuracy_score(y_test, yPred)
    tn1, fp1, fn1, tp1 = metrics.confusion_matrix(y_test, yPred).ravel()
   
    
    specificity = 0.0
    if (tn1 + fp1) != 0:   
        specificity = tn1/ (tn1 + fp1)
        
    sensitivity = 0.0
    if (tp1 + fn1) != 0:  
        sensitivity = tp1/ (tp1 + fn1)
        
    f1score = 0.0
    if (prec1 + rec1) != 0:   
        f1score = 2*((prec1*rec1)/(prec1 + rec1))
    return [float(auc1), float(prec1), float(rec1), float(acc1),float(tp1), float(tn1), float(fp1), float(fn1),float(specificity), float(sensitivity), float(f1score)]


def MLP(X, Y, targetVariable, sType, featureNames, scaleType='NoScaling', note="common"):
    rskf = RepeatedStratifiedKFold(n_splits=5,n_repeats=1, random_state=36851234)
    ind = 1
    results = []
    for train_index, test_index in rskf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
    
        #Multi-Layer deep learning Prediction
        hLayers = [30]

        initializer = tf.initializers.he_uniform()
        optimizer = 'adam'

        model, loss = predictionModel(X_train, y_train, hLayers, epochs=1000, activation='relu', kernel_initializer=initializer, optimizer=optimizer)

        #    model.summary()
        yPred = model.predict(X_test)
        sType1 = sType
        if sType1 is None:
           sType1 = "None"
        scaleType1 = scaleType
        if scaleType is None:
            scaleType1 = "None"

        algName = "MLP"
        algDetails = note + "-" + targetVariable + "-" + sType1 + "-" + scaleType1 +  "-" + algName + str(ind)
        printPdf = PdfPages(resultFolder + "/" + "ROCs/ROC" + algDetails + ".pdf")

        getROC(y_test, yPred, printPdf, algDetails = algDetails)

        printPdf.close()
        ind += 1
        res = getMetrics(y_test, yPred)
        if len(results) == 0:
            results = res
        else:
            results = np.sum((results, res), axis=0)
        print("MLP-" + scaleType1 + "," ,targetVariable, "," , sType1, "," , float(res[0]) , "," , float(res[1]) , "," , float(res[2]) , ", ", float(res[3]) , "," , float(res[4]) , "," , float(res[5]) , ", " , float(res[6]) , "," , float(res[7]) , "," , float(res[8]) , "," , float(res[9]) , "," , float(res[10]), ",", note)
  
    res = results/5
    
    print("Total-MLP-" + scaleType1 + "," ,targetVariable, "," , sType1, "," , float(res[0]) , "," , float(res[1]) , "," , float(res[2]) , ", ", float(res[3]) , "," , float(res[4]) , "," , float(res[5]) , ", " , float(res[6]) , "," , float(res[7]) , "," , float(res[8]) , "," , float(res[9]) , "," , float(res[10]), ",", note)

    
def stackedAutoEncoder(X_train, X_test, y_train, y_test, hLayers, latentDim, epochs=100, activation='relu', kernel_initializer=None, optimizer='adam', eagerly=False):

    inputDim = X_train.shape[1]
    # create model
    model = Sequential()    
    firstLayer = True

    for hSize in hLayers:
        if firstLayer:
            firstLayer = False
            if kernel_initializer:
                model.add(Dense(hSize, input_dim=inputDim, activation=activation, kernel_initializer=kernel_initializer))
            else:
                model.add(Dense(hSize,  input_dim=inputDim, activation=activation))
        else:
            if kernel_initializer:
                model.add(Dense(hSize, activation=activation, kernel_initializer=kernel_initializer))
            else:
                model.add(Dense(hSize, activation=activation))

    if kernel_initializer:
        model.add(Dense(latentDim, activation=activation, kernel_initializer=kernel_initializer))
    else:             
        model.add(Dense(latentDim, activation=activation))
   
    predModel = Sequential()
    for layer in model.layers:
        predModel.add(layer)
    predModel.add(Dense(1, activation='sigmoid'))
    
    for hSize in hLayers[::-1]:
        if kernel_initializer:
            model.add(Dense(hSize, activation=activation, kernel_initializer=kernel_initializer))
        else:
            model.add(Dense(hSize, activation=activation))
        
    if kernel_initializer:
        model.add(Dense(inputDim, activation=activation, kernel_initializer=kernel_initializer))   
    else:
        model.add(Dense(inputDim, activation=activation))   
           
     
    predModel.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'],run_eagerly=eagerly)
    
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'],run_eagerly=eagerly)
    
    loss1 = predModel.fit(X_train, y_train, epochs=epochs, batch_size=8, validation_data=(X_test, y_test), shuffle=False,validation_steps=1, verbose=0) 
    loss2 = model.fit(X_train, X_train, epochs=epochs, batch_size=8, validation_data=(X_test, X_test), shuffle=False,validation_steps=1,verbose=0) 

    return predModel, model


def sEncoder(X, Y, targetVariable, sType, featureNames, scaleType='NoScaling', note="common"):
    rskf = RepeatedStratifiedKFold(n_splits=5,n_repeats=1, random_state=36851234)

    results = []
    print("Stacked AutoEncnoder-" + scaleType + "," ,targetVariable, "," , sType)
    ind = 1
    for train_index, test_index in rskf.split(X, Y):
       
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
    

        #Multi-Layer deep learning Prediction
        hLayers = [70]
        latentDim = 30
        initializer = tf.initializers.he_uniform()
        optimizer = 'adam'  
        predModel, encModel =  stackedAutoEncoder(X_train, X_test, y_train, y_test, hLayers, latentDim, epochs=1000, activation='relu', kernel_initializer=initializer, optimizer=optimizer)
    
        #    model.summary()
#
        yPred = predModel.predict(X_test)
        sType1 = sType
        if sType1 is None:
           sType1 = "None"
        scaleType1 = scaleType
        if scaleType is None:
            scaleType1 = "None"
 
        algName = "StackedAutoEncoder"
        algDetails = note + "-" + targetVariable + "-" + sType1 + "-" + scaleType1 +  "-" + algName + str(ind)
        printPdf = PdfPages(resultFolder + "/" + "ROCs/ROC-" + algDetails + ".pdf")

        getROC(y_test, yPred, printPdf, algDetails = algDetails)
        printPdf.close()
        ind += 1
#        continue
        res = getMetrics(y_test, yPred)
        if len(results) == 0:
            results = res
        else:
            results = np.sum((results, res), axis=0)
        print("Stacked AutoEncnoder-" + scaleType1 + "," ,targetVariable, "," , sType1, "," , float(res[0]) , "," , float(res[1]) , "," , float(res[2]) , ", ", float(res[3]) , "," , float(res[4]) , "," , float(res[5]) , ", " , float(res[6]) , "," , float(res[7]) , "," , float(res[8]) , "," , float(res[9]) , "," , float(res[10]), ",", note)
    

    res = results/5
    print("Total-Stacked AutoEncnoder-" + scaleType1 + "," ,targetVariable, "," , sType1, "," , float(res[0]) , "," , float(res[1]) , "," , float(res[2]) , ", ", float(res[3]) , "," , float(res[4]) , "," , float(res[5]) , ", " , float(res[6]) , "," , float(res[7]) , "," , float(res[8]) , "," , float(res[9]) , "," , float(res[10]), ",", note)

def genAdversarialNetwork(X, Y, latentDim=50, descLayers = [50], genLayers = [70],latentLayers = [70], kernel_initializer=None, dActivation='relu', gActivation='relu', lActivation='relu', epochs=1000, optimizer='adam', verbose=2):

    inputDim = X.shape[1]

    # create model
    discModel = Sequential()    
    firstLayer = True

    for hSize in descLayers:
        if firstLayer:
            firstLayer = False
            if kernel_initializer:
                discModel.add(Dense(hSize, input_dim=inputDim, activation=dActivation, kernel_initializer=kernel_initializer))
            else:
                discModel.add(Dense(hSize, input_dim=inputDim, activation=dActivation))
        else:
            if kernel_initializer:
                discModel.add(Dense(hSize, activation=dActivation, kernel_initializer=kernel_initializer))
            else:
                discModel.add(Dense(hSize, activation=dActivation))

    discModel.add(Dense(1, activation='sigmoid'))

    # create model
    genModel = Sequential()    
    firstLayer = True

    for hSize in latentLayers:
        if firstLayer:
            firstLayer = False
            if kernel_initializer:
                genModel.add(Dense(hSize, input_dim=inputDim, activation=lActivation, kernel_initializer=kernel_initializer))
            else:
                genModel.add(Dense(hSize, input_dim=inputDim, activation=lActivation))
        else:
            if kernel_initializer:
                genModel.add(Dense(hSize, activation=lActivation, kernel_initializer=kernel_initializer))
            else:
                genModel.add(Dense(hSize, activation=lActivation))

    if kernel_initializer:
        genModel.add(Dense(latentDim, activation=lActivation, kernel_initializer=kernel_initializer))
    else:             
        genModel.add(Dense(latentDim, activation=lActivation))
   
    for hSize in genLayers:
        if kernel_initializer:
            genModel.add(Dense(hSize, activation=gActivation, kernel_initializer=kernel_initializer))
        else:
            genModel.add(Dense(hSize, activation=gActivation))
        
    if kernel_initializer:
        genModel.add(Dense(inputDim, activation='linear', kernel_initializer=kernel_initializer))   
    else:
        genModel.add(Dense(inputDim, activation='linear')) 
        
    
    genModel.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    discModel.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])    
       

    xNeg = []
    yNeg = []
    for ind, yVal in enumerate(Y):
        if yVal == 1:
            yNeg.append(1)
            xNeg.append(X[ind])          
    
    xNeg = np.asarray(xNeg)
    yNeg = np.asarray(yNeg)    
    
    genLoss = genModel.fit(xNeg, xNeg, epochs=epochs, batch_size=8, validation_data=(xNeg, xNeg), shuffle=False,validation_steps=1, verbose=verbose) 
    xPredNeg = genModel.predict(xNeg)
    
    ganLoss = discModel.fit(xPredNeg, yNeg, epochs=epochs, batch_size=8, validation_data=(xPredNeg, yNeg), shuffle=False,validation_steps=1, verbose=verbose)  

    dLoss = discModel.fit(X, Y, epochs=epochs, batch_size=8, validation_data=(X, Y), validation_steps=1, shuffle=False,verbose=verbose)  

    return discModel, dLoss

def ganCall(X, Y, targetVariable, sType, featureNames, scaleType='NoScaling', note="common"):
    rskf = RepeatedStratifiedKFold(n_splits=5,n_repeats=1, random_state=36851234)
    print("GAN-" + scaleType + "," ,targetVariable, "," , sType)
    ind = 1
    results = []
    for train_index, test_index in rskf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        #Multi-Layer deep learning Prediction
        hLayers = [30]
        initializer = tf.initializers.he_uniform()
        optimizer = 'adam'
        
        model, loss = genAdversarialNetwork(X_train, y_train, latentDim=50, descLayers = [50], genLayers = [70],latentLayers = [70], kernel_initializer=initializer, dActivation='relu', gActivation='relu', lActivation='relu', epochs=1000, optimizer=optimizer, verbose=0)

        #    model.summary()
        yPred = model.predict(X_test)
        sType1 = sType
        if sType1 is None:
           sType1 = "None"
        scaleType1 = scaleType
        if scaleType is None:
            scaleType1 = "None"

        algName = "GAN"
        algDetails = note + "-" + targetVariable + "-" + sType1 + "-" + scaleType1 +  "-" + algName + str(ind)
        printPdf = PdfPages(resultFolder + "/" + "ROCs/ROC-" + algDetails + ".pdf")

        getROC(y_test, yPred, printPdf, algDetails = algDetails)

        printPdf.close()
        ind += 1
  
        #    model.summary()
        res = getMetrics(y_test, yPred)
        if len(results) == 0:
            results = res
        else:
            results = np.sum((results, res), axis=0)
        print("GAN-" + scaleType1 + "," ,targetVariable, "," , sType1, "," , float(res[0]) , "," , float(res[1]) , "," , float(res[2]) , ", ", float(res[3]) , "," , float(res[4]) , "," , float(res[5]) , ", " , float(res[6]) , "," , float(res[7]) , "," , float(res[8]) , "," , float(res[9]) , "," , float(res[10]), ",", note)

    res = results/5

    print("Total-GAN-" + scaleType1 + "," ,targetVariable, "," , sType1, "," , float(res[0]) , "," , float(res[1]) , "," , float(res[2]) , ", ", float(res[3]) , "," , float(res[4]) , "," , float(res[5]) , ", " , float(res[6]) , "," , float(res[7]) , "," , float(res[8]) , "," , float(res[9]) , "," , float(res[10]), ",", note)  
   

def createFolders():
    for folderName in ['ROCs', 'FeatureImportantResults']:
        if not os.path.exists(resultFolder  + "/" + folderName):
            os.makedirs(resultFolder + "/" + folderName)  

def mlModels(algName, weights):
    model = None
    if algName == 'svm-balanced':
        # SVM for balanced data
        model = SVC(gamma='scale')
    elif algName == 'svm-weighted':        
        # SVM for imbalanced data
        model = SVC(gamma='scale', class_weight=weights)
    elif algName == 'lr-balanced': 
        # LR for balanced data
        model = LogisticRegression()
    elif algName == 'lr-weighted': 
        # LR for imbalanced data        
        model = LogisticRegression(solver='lbfgs', class_weight=weights)
    elif algName == 'xgboost-weighted': 
        # XGBoost from imbalanced data
        model = XGBClassifier(scale_pos_weight=weights[1],use_label_encoder=False, eval_metric='logloss')
    elif algName == 'easyensemble-weighted': 
        # Easy Ensemble for imbalanced data
        model = EasyEnsembleClassifier(n_estimators=10)
    elif algName == 'randomforest-weighted': 
        # BalancedRandomForestClassifier for imbalanced data
        model = BalancedRandomForestClassifier(n_estimators=100)
    return model

def summaryPlot(X, model, featureNames, show=True, title=None, shapplot=True):
# select a set of background examples to take an expectation over
    background = X[np.random.choice(X.shape[0], 100, replace=True)]

    explainer = shap.DeepExplainer(model, background)

    shap_values = explainer.shap_values(X) # <-- HERE ! 
    if shapplot:
       shap.summary_plot(shap_values, X, featureNames, plot_type="bar", plot_size=None, title=title, show=show)  
    return shap_values[0]
 
    
def sklearnSummaryPlot(dl, X, model, featureNames, show=True, title=None,shapplot=True):
    from os.path import exists
    import joblib
    import pickle

    shap_values, plot_values = None, None   
    if dl == "rt":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X, approximate=False, check_additivity=False)
        plot_values = shap_values[0]
    elif dl == 'xgb':
        xgb_explainer = shap.TreeExplainer(model)
        shap_values = xgb_explainer.shap_values(X, approximate=False, check_additivity=False)

        plot_values = shap_values
    if shapplot:
       shap.summary_plot(plot_values, X, featureNames, plot_type="bar", plot_size=None, title=title, show=show)
    return plot_values

def getShapPlot(sortedFeatures, dl, nFolder, algFName, ind, featureNames, shap_values, X, top):
   for rank in range(top):
      printPdf = PdfPages(nFolder + "/" + dl + str(ind) +  "-rank" + str(rank + 1) +  ".pdf")
      plt.ioff()
      fig = plt.figure()
      plt.rc('axes', linewidth=1)
      plt.rc('font', weight='normal')
      lw = 2
      plt.tight_layout()
      fSize = 20
      plt.rcParams["font.family"] = "Times New Roman"
      plt.rcParams["font.size"] = fSize
      plt.rcParams['axes.labelsize'] = fSize
      axes = plt.gca()
      axes.xaxis.label.set_size(fSize)
      axes.yaxis.label.set_size(fSize)
      algDetails = algFName + "-" + dl + str(ind) +  "-rank" + str(rank + 1)
#      dependancePlot(shapValues[dl+str(ind)], xValues[dl+str(ind)], featureNames, show=False, rank=rank + 1, shapplot=False, algDetails=algDetails)

      result = np.where(featureNames == sortedFeatures[-1 * (rank + 1)])
      rankInd = result[0][0]
      plt.scatter(X[:,rankInd], shap_values[:,rankInd])
      fSize = 20
      plt.xlabel(featureNames[rankInd], fontsize=fSize)
      plt.ylabel("Prediction Relevance", fontsize=fSize)

      plt.savefig(printPdf, format='pdf', bbox_inches='tight',pad_inches=0.5, transparent=True)
      printPdf.close()
      plt.close(fig)

    
def barplot(dl, nFolder, algDetails, indX, shap_values, X, names, top=10):
   fSize = 30
   featureNames = names
   values = np.mean(shap_values, axis=0)
   
   df = pd.DataFrame ({ 'Group': names, 'Value' : values })
   df = df.sort_values(by=['Value'])
   df = df.iloc[-1 * top :]

   vMin = float("%.2f" % np.min(df['Value']))
   vMax = float("%.2f" % np.max(df['Value']))
   tickMid = float((vMax + vMin) / 2.0)
   ticks = [vMin, tickMid, vMax]

   print(",".join([str(val) for val in list(df['Group'])[::-1]]))
#   print(",".join([str(val) for val in list(df['Value'])[::-1]]))

   getShapPlot([str(val) for val in list(df['Group'])], dl, nFolder, algDetails, indX, featureNames, shap_values, X, top)
   return [str(val) for val in list(df['Group'])]



def getFeatureImportance(X, Y, dlTypes, targetVariable, sType,  featureNames, scaleType, note="common"):
        shapValues = {}
        xValues = {}
        plt.ioff()

        algName = None
        import os
        nFolder = resultFolder + "/" + "FeatureImportantResults/" + note + "-" + targetVariable + "-" + sType + "-" + scaleType
        os.system("mkdir -p " + nFolder)
        fSize = 30

        plt.rcParams['axes.labelsize'] = fSize
        plt.rc('axes', linewidth=4)
        plt.rc('font', weight='bold')
        rskf = RepeatedStratifiedKFold(n_splits=5,n_repeats=1, random_state=36851234)
        indX = 1
        results = []
        for train_index, test_index in rskf.split(X, Y):
           X_train, X_test = np.array(X[train_index]), np.array(X[test_index])
           y_train, y_test = np.array(Y[train_index]), np.array(Y[test_index])
           for ind, dl in enumerate(dlTypes):
                title = ""
                model = None
                if dl == "mlp":
                        hLayers = [30]
                        initializer = tf.initializers.he_uniform()
                        optimizer = 'adam'
                        model, loss = predictionModel(X_train, y_train, hLayers, epochs=1000, activation='relu', kernel_initializer=initializer, optimizer=optimizer)
                        
                        title = "Multi-LayerPerceptron"
                elif dl == "sEncoding": 

                        hLayers = [70]
                        latentDim = 30
                        initializer = tf.initializers.he_uniform()
                        optimizer = 'adam'  
                        model, encModel =  stackedAutoEncoder(X_train, X_test, y_train, y_test, hLayers, latentDim, epochs=1000, activation='relu', kernel_initializer=initializer, optimizer=optimizer, eagerly=False)

                        title = "StackedAutoEncoder"
                elif dl == "gan":
                        hLayers = [30]
                        initializer = tf.initializers.he_uniform()
                        optimizer = 'adam'     
                        model, loss = genAdversarialNetwork(X_train, y_train, latentDim=50, descLayers = [50], genLayers = [70],latentLayers = [70], kernel_initializer=initializer, dActivation='relu', gActivation='relu', lActivation='relu', epochs=1000, optimizer=optimizer, verbose=0)

                        title = "GAN"
                elif dl == "rt":
                        algName = "randomforest-weighted"
                        title = "RandomForest"
                elif dl == "xgb":
                        algName = "xgboost-weighted"
                        title = "XGBoost"

                plt.subplot(1, len(dlTypes), ind+1)

                shap_values = None
                if dl == "rt" or dl == "xgb":
                        model = None
                        weight0 = int(float((len(Y) - sum(Y))/len(Y)) * 100)
                        weight1 = 100 - weight0
                        weights = {0:weight1, 1:weight0}
                        model = mlModels(algName, weights)
                        model.fit(X_train, y_train)
                        
                        shap_values = sklearnSummaryPlot(dl, X_test, model, featureNames, show=False, title=title, shapplot=False)
                else:
                        shap_values  = summaryPlot(X_test, model, featureNames, show=False, title=title, shapplot=False)

#                import tensorflow as tf
#                tf.compat.v1.disable_eager_execution()
                plt.rcParams['axes.labelsize'] = fSize
                plt.rc('axes', linewidth=4)
                plt.rc('font', weight='bold')

                plt.rcParams["font.family"] = "Times New Roman"
                algDetails = note + "-" + targetVariable + "-" + sType + "-" + scaleType 
                print(dl, " - Stratified Fold " , str(indX), " Top 15 Features::")
                barplot(dl, nFolder, algDetails, indX, shap_values, X_test, featureNames, top=20)

                shapValues[dl + str(indX)] = shap_values
                xValues[dl + str(indX)] = X_test
           indX += 1


