import lib
from lib import *
import tensorflow as tf
#tf.enable_eager_execution()


import numpy as np
import os
import argparse

from imblearn.over_sampling import SMOTE

tf.get_logger().setLevel('INFO')

tf.compat.v1.disable_v2_behavior() # <-- HERE !

parser = argparse.ArgumentParser()

parser.add_argument("-t", "--Target", help = "MDR", choices=["C2_MDR","L_MDR","S_MDR"], required=True)
parser.add_argument("-d", "--Dataset", help = "Dataset in CSV format", required=True)

args = parser.parse_args()
poultryFileName = args.Dataset
targetVariable = args.Target

print(poultryFileName,targetVariable)

#poultryFileName = "Pastured_Poultry_Summary_DB_03162020_PrePostHarvestModifiedVariables.csv"

#targetVariable = "c2_mdr"


targetVariable = targetVariable.upper()

commonVariables = ['AvgNumBirds','AvgNumFlocks','YearsFarming','EggSource','BroodBedding','BroodFeed','BrGMOFree','BrSoyFree', 'BroodCleanFrequency','AvgAgeToPasture','PastureHousing','FreqHousingMove','AlwaysNewPasture','PastureFeed','PaGMOFree','PaSoyFree','LayersOnFarm','CattleOnFarm','SwineOnFarm','GoatsOnFarm','SheepOnFarm','WaterSource','FreqBirdHandling','AnyABXUse','FlockAgeDays', 'Breed','FlockSize','SampleType','AnimalSource']
preVariables = ['pH','EC','Moisture','TotalC','TotalN','CNRatio','Ca','Cd','Cr','Cu','Fe','K','Mg','Mn','Mo','Na','Ni','P','Pb','Zn']

independantVariables = commonVariables + preVariables

sampleTypes = ['Feces', 'Soil','pre-harvest']

note = "pre"

lib.resultFolder = "FeatureImportance_" + str(targetVariable) + "/"

oversamplers = ["smote"]
dlTypes = ['rt',"xgb",'mlp', 'sEncoding', 'gan']
 
checkRequiredColumns(poultryFileName,independantVariables,targetVariable) 
createFolders()

for oversampling in oversamplers:  
    for sType in sampleTypes:
        dfNotNaN, binaryVars, categoryVars, numericVars = prepareData(poultryFileName, independantVariables, [targetVariable], sampleTypes)
        X, Y = prepareXY(dfNotNaN, sType, binaryVars, categoryVars, numericVars, [targetVariable])
        scaleType1 = 'NoScaling'
        
        featureNames = X.columns
        if np.sum(Y) == 0 or np.sum(Y) == len(Y):
            continue

                    
        X = np.array(X)
                        

        # transform the dataset
        oversample = SMOTE(k_neighbors=1)
        X, Y = oversample.fit_resample(X, Y)
        scaleType1 += '-SmoteOversampling'

           
 
        getFeatureImportance(X, Y, dlTypes, targetVariable, sType,  featureNames, scaleType1, note=note)
    


