from lib import *
import lib
import tensorflow as tf
tf.executing_eagerly()

import argparse

import numpy as np
import os
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler

tf.get_logger().setLevel('INFO')

parser = argparse.ArgumentParser()

parser.add_argument("-t", "--Target", help = "MDR", choices=["C2_MDR","L_MDR","S_MDR"], required=True)
parser.add_argument("-d", "--Dataset", help = "Dataset in CSV format", required=True)

args = parser.parse_args()

poultryFileName = args.Dataset
targetVariable = args.Target

#poultryFileName = "Pastured_Poultry_Summary_DB_03162020_PrePostHarvestModifiedVariables.xlsx"

#targetVariable = "c2_mdr"


targetVariable = targetVariable.upper()

commonVariables = ['AvgNumBirds','AvgNumFlocks','YearsFarming','EggSource','BroodBedding','BroodFeed','BrGMOFree','BrSoyFree', 'BroodCleanFrequency','AvgAgeToPasture','PastureHousing','FreqHousingMove','AlwaysNewPasture','PastureFeed','PaGMOFree','PaSoyFree','LayersOnFarm','CattleOnFarm','SwineOnFarm','GoatsOnFarm','SheepOnFarm','WaterSource','FreqBirdHandling','AnyABXUse','FlockAgeDays', 'Breed','FlockSize','SampleType','AnimalSource']
preVariables = ['pH','EC','Moisture','TotalC','TotalN','CNRatio','Ca','Cd','Cr','Cu','Fe','K','Mg','Mn','Mo','Na','Ni','P','Pb','Zn']

independantVariables = commonVariables + preVariables

sampleTypes = ['Feces', 'Soil','pre-harvest']
note = "pre"

lib.resultFolder = "Results_" + str(targetVariable) + "/"


dl = ""
scaleType = ""
oversampling = ""

if targetVariable == "L_MDR":
    dl = 'gan'
    scaleType = "robustScale"
    oversampling = 'smote'
elif targetVariable == "S_MDR":
    dl = 'sEncoding'
    scaleType = "quantileTranform"
    oversampling = 'random'
elif targetVariable == "C2_MDR":
    dl = 'mlp'
    scaleType = "unitnorm"
    oversampling = 'smote'

checkRequiredColumns(poultryFileName,independantVariables,targetVariable)  
createFolders()

print(",,,AUC, Precision, Recall, Accuracy, TP, TN, FP, FN, Specificity, Sensitivity, F1score")

for sType in sampleTypes:
    dfNotNaN, binaryVars, categoryVars, numericVars = prepareData(poultryFileName, independantVariables, [targetVariable], sampleTypes)
    X, Y = prepareXY(dfNotNaN, sType, binaryVars, categoryVars, numericVars, [targetVariable])  
    featureNames = X.columns
    if np.sum(Y) == 0 or np.sum(Y) == len(Y):
            continue

    scaleType1 = scaleType
    X = np.array(X)
    X = scaler(X, scaleType)

                        
    if oversampling == "smote":
        # transform the dataset
        oversample = SMOTE(k_neighbors=1)
        X, Y = oversample.fit_resample(X, Y)
        scaleType1 += '-SmoteOversampling'
    elif oversampling == "random":
        oversample = RandomOverSampler(random_state=42)
        X, Y = oversample.fit_resample(X, Y)
        scaleType1 += '-RandomOversampling'
    if dl == 'mlp':
        MLP(X, Y, targetVariable, sType,  featureNames, scaleType1, note=note)
    elif dl == 'sEncoding': 
        sEncoder(X, Y, targetVariable, sType, featureNames, scaleType1, note=note)
    elif dl == 'gan':
        ganCall(X, Y, targetVariable, sType, featureNames, scaleType1, note=note)

                
