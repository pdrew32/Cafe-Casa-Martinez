import numpy as np
import pandas as pd
import constants_helper as c
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle


"""
run classification algorithms to determine the best one and save the trained model for use in other scripts
"""


save_coefs = True
best_fit_coef_path = 'data/lr_coefs_best_fit.pkl'

# load the data
tot_plants = pd.read_csv('data/tot_plants.csv', index_col=0)
# define the profit threshold and make a new binary column where 1 is a year with profit and 0 is without
profit_threshold = c.c.profit_thresh_kg
tot_plants['profit'] = ((tot_plants['total_production'] > profit_threshold)*1).values

# get only columns to perform logistic regression on
use_cols = ['total_rain_cm', 'jan_rain_cm', 'feb_rain_cm', 'mar_rain_cm',
       'apr_rain_cm', 'may_rain_cm', 'jun_rain_cm', 'jul_rain_cm',
       'aug_rain_cm', 'sep_rain_cm', 'oct_rain_cm', 'nov_rain_cm',
       'dec_rain_cm']
X = np.asarray(tot_plants[use_cols])
y = np.asarray(tot_plants['profit'])

# compute anova f statistic to find which input variables correlate most strongly with the output variable
f_stat, p_val = f_classif(X, y)
print('months with rainfall correlating with profit:', np.array(use_cols)[p_val < 0.05])

# select K variables that correlate most strongly with binary profit column
fs = SelectKBest(score_func=f_classif, k=len(np.array(use_cols)[p_val < 0.05]))
# reduce X to X_selected, the most important input variables
X_selected = fs.fit_transform(X, y)

# create standardizer
standardizer = StandardScaler()

# create model
lr = LogisticRegression(solver='lbfgs')
gnb = GaussianNB()
knn = KNeighborsClassifier(n_neighbors=3)
sv = svm.SVC()
tr = tree.DecisionTreeClassifier()
rf = RandomForestClassifier()

# create a pipeline to standardize and then run the model
pipeline_lr = make_pipeline(standardizer, lr)
pipeline_gnb = make_pipeline(standardizer, gnb)
pipeline_knn = make_pipeline(standardizer, knn)
pipeline_svm = make_pipeline(standardizer, sv)
pipeline_tr = make_pipeline(standardizer, tr)
pipeline_rf = make_pipeline(standardizer, rf)

param_grid = {'svc__C': [0.1, 1, 10, 100, 1000],
              'svc__gamma': ['scale', 'auto'],
              'svc__kernel': ['linear', 'rbf'],
              'svc__degree': [1, 2, 3, 4, 5]}
 
grid = GridSearchCV(pipeline_svm, param_grid, refit = True)
grid.fit(X_selected, y)

print(grid.best_params_)

sv = svm.SVC(C = grid.best_params_['svc__C'], degree=grid.best_params_['svc__degree'], gamma=grid.best_params_['svc__gamma'], kernel=grid.best_params_['svc__kernel'])
pipeline_svm = make_pipeline(standardizer, sv)

# use stratified k fold cross validation because the split of non-profit to profit is about 0.7 to 0.3
n_splits = 4
skf = StratifiedKFold(n_splits=n_splits)

cv_results_lr = cross_val_score(pipeline_lr, X_selected, y, scoring='accuracy', cv=skf)
print('cross validation logistic regression mean:', cv_results_lr.mean())

cv_results_gnb = cross_val_score(pipeline_gnb, X_selected, y, scoring='accuracy', cv=skf)
print('cross validation gaussian naive bayes mean:', cv_results_gnb.mean())

cv_results_knn = cross_val_score(pipeline_knn, X_selected, y, scoring='accuracy', cv=skf)
print('cross validation k nearest neighbors:', cv_results_knn.mean())

cv_results_svm = cross_val_score(pipeline_svm, X_selected, y, scoring='accuracy', cv=skf)
print('cross validation support vector machine:', cv_results_svm.mean())

cv_results_tr = cross_val_score(pipeline_tr, X_selected, y, scoring='accuracy', cv=skf)
print('cross validation decision tree:', cv_results_tr.mean())

cv_results_rf = cross_val_score(pipeline_rf, X_selected, y, scoring='accuracy', cv=skf)
print('cross validation random forest:', cv_results_rf.mean())

pipeline_lr.fit(X_selected, y)
pipeline_gnb.fit(X_selected, y)
pipeline_knn.fit(X_selected, y)
pipeline_svm.fit(X_selected, y)

print('logistic regression')
print(classification_report(y, pipeline_lr.predict(X_selected)))
print('gaussian naive bayes')
print(classification_report(y, pipeline_gnb.predict(X_selected)))
print('k nearest neighbors')
print(classification_report(y, pipeline_knn.predict(X_selected)))
print('support vector machine')
print(classification_report(y, pipeline_svm.predict(X_selected)))

# fit to all the data
lr.fit(X_selected, y)
sv.fit(X_selected, y)

print('logistic regression')
print(confusion_matrix(y, lr.predict(X_selected)))
print('support vector machine')
print(confusion_matrix(y, sv.predict(X_selected)))

if save_coefs is True:
       with open(best_fit_coef_path, 'wb') as file:
              pickle.dump(lr, file)