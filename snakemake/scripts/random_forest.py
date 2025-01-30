import pandas as pd
import numpy as np
# scikit-learn version: 1.5.1
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score, get_scorer_names
from sklearn.utils import check_X_y
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict, cross_validate, GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from skl2onnx import to_onnx

#  function to add the random_state to the mutual information function
def featureSelectionScoringFunction(X, y):
    return mutual_info_classif(X, y, random_state = 0)

np.random.seed(0)

# training data
features = pd.read_csv(snakemake.input[0], delimiter = ','); # features
labels = pd.read_csv(snakemake.input[2], delimiter = ','); # label

# test data
featuresTest = pd.read_csv(snakemake.input[1], delimiter = ',');
labelsTest = pd.read_csv(snakemake.input[3], delimiter = ',');

if(snakemake.wildcards.ts == 'noTs'):
    # training data
    featuresBefore = pd.read_csv(snakemake.input[0], delimiter = ','); # features
    labelsBefore = pd.read_csv(snakemake.input[2], delimiter = ','); # label

    # test data
    featuresTestBefore = pd.read_csv(snakemake.input[1], delimiter = ',');
    labelsTestBefore = pd.read_csv(snakemake.input[3], delimiter = ',');

    features = featuresBefore.drop(['Unnamed: 0'], axis = 1)
    featuresTest = featuresTestBefore.drop(['Unnamed: 0'], axis = 1)
    labels = labelsBefore.drop(['Unnamed: 0', 'Unnamed: 0.1'],  axis = 1)
    labelsTest = labelsTestBefore.drop(['Unnamed: 0', 'Unnamed: 0.1'],  axis = 1)


pTraining = features['pseudonyms']
pTest = featuresTest['pseudonyms']

# drop columns that should not be used by the random forest
X = features.drop(['pseudonyms'], axis=1)
XTest = featuresTest.drop(['pseudonyms'], axis=1)
y = labels.label
yTest = labelsTest.label
X_column_names = X.columns

X = X.fillna(-1000)
X, y = check_X_y(X,y)

XTest = XTest.fillna(-1000)
XTest, yTest = check_X_y(XTest,yTest)

hyperparameters = {}

# Random Forest Hyperparameters
if(snakemake.config['bootstrap'] == True):

    maxSamples = []

    if(len(snakemake.config['maxSamples']) == 1 and snakemake.config['maxSamples'][0] == 'None'):
        maxSamples = [None]
    else:
        maxSamples = snakemake.config['maxSamples']

    hyperparameters = {
    'model__max_depth': [2],
    'model__max_features': ['log2', 'sqrt'],
    'model__min_samples_leaf': [2, 4, 7],
    'model__class_weight' : ['balanced'],
    'model__random_state': [0],
    'model__n_estimators': [100, 150, 200, 250],
    'model__n_jobs': [snakemake.threads],
    'model__bootstrap': [True],
    'model__max_samples': maxSamples,
    'selector__k': [20, 40, 100, 200, len(X_column_names)],
    }
else:
    hyperparameters = {
    'model__max_depth': [2],
    'model__max_features': ['log2', 'sqrt'],
    'model__min_samples_leaf': [2, 4, 7],
    'model__class_weight' : ['balanced'],
    'model__random_state': [0],
    'model__n_estimators': [100, 150, 200, 250],
    'model__n_jobs': [snakemake.threads],
    'model__bootstrap': [False],
    'model__max_samples': [None],
    'selector__k': [20, 40, 100, 200, len(X_column_names)],
    }
rf_cv = RandomForestClassifier(random_state = 0);

# pipeline with feature selection and random forest
pipeline_rf = Pipeline([
    ('selector', SelectKBest(score_func = featureSelectionScoringFunction)),
    ('model', rf_cv)
])

# use this to ensure the folds are stratified
#cvInner = StratifiedKFold(n_splits = 5, shuffle = False, random_state = None) # random_state only affects the function if shuffle is True; returns an error if set while shuffle = False
cvInner = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 5, random_state = 0)

################ CV (NOT nested) ####################
if(snakemake.wildcards.version == '1'):
    rf_grid_cv = GridSearchCV(estimator = pipeline_rf, scoring= 'f1', refit = True, param_grid = hyperparameters, cv = cvInner, n_jobs = snakemake.threads, verbose = 3)
    #rf_grid_cv = RandomizedSearchCV(estimator = pipeline_rf, scoring= 'f1', refit = True, param_distributions = hyperparameters, n_iter = 5, cv = cvInner, n_jobs = snakemake.threads, verbose = 0, random_state = 0)
    rf_grid_cv.fit(X, y)

    # write best parameter combination into a file
    bestParams = rf_grid_cv.best_params_
    bestParamsDf = pd.DataFrame(bestParams, index = [0])
    bestParamsDf.to_csv(snakemake.output[3])

    # write all parameter combinations with scores into a file
    allScoresGrid = rf_grid_cv.cv_results_ # returns a dictionary
    dfAllScoresGrid = pd.DataFrame.from_dict(allScoresGrid)
    dfAllScoresGrid.to_csv(snakemake.output[6])


    ## feature selection
    feature_selectorRF = SelectKBest(score_func = featureSelectionScoringFunction, k=bestParams['selector__k'])
    X_k_rf = feature_selectorRF.fit_transform(X, y)
    selectedColumnsRF = feature_selectorRF.get_support()

    indicesSelectedColumnsRF = np.where(selectedColumnsRF)[0]
    namesSelectedColumnsRF = [X_column_names[i] for i in indicesSelectedColumnsRF]

    # select test data according to the columns selected for the training data
    X_k_test_rf = XTest[:, indicesSelectedColumnsRF]

    # fit **final** random forest classifier with the best hyperparameters and the features selected above
    rf = RandomForestClassifier(n_jobs = snakemake.threads,bootstrap = bestParams['model__bootstrap'], max_samples = bestParams['model__max_samples'], n_estimators = bestParams['model__n_estimators'], max_features = bestParams['model__max_features'], max_depth = bestParams['model__max_depth'], min_samples_leaf=bestParams['model__min_samples_leaf'], random_state = 0, class_weight='balanced')
    rf.fit(X_k_rf, y)

    # write feature importances of trained random forest to file
    importances = pd.DataFrame(data={
        'Importance': rf.feature_importances_,
        'Names': namesSelectedColumnsRF
    })
    importances.to_csv(snakemake.output[1]);

    # feature importances sorted 
    importancesSorted = importances.sort_values(by='Importance', ascending=False)
    importancesSorted.to_csv(snakemake.output[2]);

    ## save rf model
    saveRf = to_onnx(rf, X_k_rf[:1].astype(np.float32), target_opset=12)
    with open(snakemake.output[9], "wb") as f:
        f.write(saveRf.SerializeToString())

    ###### random forest test data predictions ######

    predictionProbabilityRf = np.around(rf.predict_proba(X_k_test_rf)[:,1], decimals = 12)

    # AUROC calculations
    aurocRf = roc_auc_score(yTest, predictionProbabilityRf)
    FPR_RF, TPR_RF, thresholds_RF = roc_curve(yTest, predictionProbabilityRf, pos_label=1)

    dfRocRF = pd.DataFrame({'FPR': FPR_RF, 'TPR': TPR_RF, 'Thresholds': thresholds_RF})
    dfRocRF.to_csv(snakemake.output[4])

    # get adjusted threshold
    index_RF = -1
    for fpr in FPR_RF:
        if(fpr <= 0.2):
            index_RF = index_RF + 1

    # prediction with adjusted threshold
    predictionRf = (np.around(rf.predict_proba(X_k_test_rf)[:, 1], decimals = 12) >= thresholds_RF[index_RF]).astype(int)

    # write predictions to file with pseudonyms
    patientOutputTest = pd.DataFrame({'pseudonyms': pTest, 'probabilities': predictionProbabilityRf, 'binary': predictionRf})
    patientOutputTest.to_csv(snakemake.output[7])

    # metrics for the test data
    accuracyRf = accuracy_score(yTest, predictionRf)
    accuracy_countRf = accuracy_score(yTest, predictionRf, normalize=False)

    confusionMRf = confusion_matrix(yTest, predictionRf)
    recallRf = recall_score(yTest, predictionRf)
    precisionRf = precision_score(yTest, predictionRf)
    f1scoreRf = f1_score(yTest, predictionRf)

    # AUPRC calculations
    precisionC, recallC, thresholdsC = precision_recall_curve(yTest, predictionProbabilityRf)
    auprcRf = auc(recallC, precisionC)

    dfPrcRF = pd.DataFrame({'Precision': precisionC[0:len(precisionC)-1], 'Recall': recallC[0:len(recallC)-1], 'Thresholds': thresholdsC})
    dfPrcRF.to_csv(snakemake.output[5])


    ###### random forest training data prediction ######

    # use the same threshold as for the test data
    predictionTrainingRf =(np.around(rf.predict_proba(X_k_rf)[:, 1], decimals = 12) >= thresholds_RF[index_RF]).astype(int)
    predictionProbabilityTrainingRf = np.around(rf.predict_proba(X_k_rf)[:,1], decimals = 12)

    # ROC for the training data
    FPR_Trainig_RF, TPR_Training_RF, thresholds_Training_RF = roc_curve(y, predictionProbabilityTrainingRf, pos_label=1)

    dfRocTrainingRF = pd.DataFrame({'FPR': FPR_Trainig_RF, 'TPR': TPR_Training_RF, 'Thresholds': thresholds_Training_RF})
    dfRocTrainingRF.to_csv(snakemake.output[10])

    # write training predictions to file with pseudonyms
    patientOutputTraining = pd.DataFrame({'pseudonyms': pTraining, 'probabilities': predictionProbabilityTrainingRf, 'binary': predictionTrainingRf})
    patientOutputTraining.to_csv(snakemake.output[8])

    # metrics for the training data
    aurocTRf = roc_auc_score(y, predictionProbabilityTrainingRf)
    accuracyTRf = accuracy_score(y, predictionTrainingRf)
    accuracy_countTRf = accuracy_score(y, predictionTrainingRf, normalize=False)

    confusionMTRf = confusion_matrix(y, predictionTrainingRf)
    recallTRf = recall_score(y, predictionTrainingRf)
    precisionTRf = precision_score(y, predictionTrainingRf)
    f1scoreTRf = f1_score(y, predictionTrainingRf)

    # AUPRC Score
    precisionTemp, recallTemp, thresholdsTemp = precision_recall_curve(y, predictionProbabilityTrainingRf)
    auprcTRf = auc(recallTemp, precisionTemp)

    dfPrcTrainingRF = pd.DataFrame({'Precision': precisionTemp[0:len(precisionTemp)-1], 'Recall': recallTemp[0:len(recallTemp)-1], 'Thresholds': thresholdsTemp})
    dfPrcTrainingRF.to_csv(snakemake.output[11])


################# save all metrics to a csv file #####################

if(snakemake.wildcards.version == '1'):
    metricsDict = {'accuracyTest': accuracyRf, 'accuracyNotNormalizedTest': accuracy_countRf, 'confusionMatrixTopLeftTest': confusionMRf[0][0], 'confusionMatrixTopRightTest': confusionMRf[0][1], 'confusionMatrixBottomLeftTest': confusionMRf[1][0], 'confusionMatrixBottomRightTest': confusionMRf[1][1], 'recallTest': recallRf, 'precisionTest': precisionRf, 'f1Test': f1scoreRf, 'aurocTest': aurocRf, 'auprcTest': auprcRf, 'accuracyTraining': accuracyTRf, 'accuracyNotNormalizedTraining': accuracy_countTRf, 'confusionMatrixTopLeftTraining': confusionMTRf[0][0], 'confusionMatrixTopRightTraining': confusionMTRf[0][1], 'confusionMatrixBottomLeftTraining': confusionMTRf[1][0], 'confusionMatrixBottomRightTraining' :confusionMTRf[1][1], 'recallTraining': recallTRf, 'precisionTraining': precisionTRf, 'f1Training': f1scoreTRf, 'aurocTraining': aurocTRf, 'auprcTraining': auprcTRf}

    metricsDf = pd.DataFrame(metricsDict, index= [0])
    metricsDf.to_csv(snakemake.output[0])
