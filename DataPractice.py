# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:34:11 2017

@author: Eugene
"""

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LassoLarsCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
 

# Modelling Helpers
from sklearn.model_selection import cross_val_score,GridSearchCV,ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import make_scorer,accuracy_score
from sklearn.preprocessing import Imputer , Normalizer , scale, StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.externals.joblib import parallel_backend

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Configure visualisations
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6
def plot_histograms( df , variables , n_rows , n_cols ):
    fig = plt.figure( figsize = ( 16 , 12 ) )
    for i, var_name in enumerate( variables ):
        ax=fig.add_subplot( n_rows , n_cols , i+1 )
        df[ var_name ].hist( bins=10 , ax=ax )
        ax.set_title( 'Skew: ' + str( round( float( df[ var_name ].skew() ) , ) ) ) # + ' ' + var_name ) #var_name+" Distribution")
        ax.set_xticklabels( [] , visible=False )
        ax.set_yticklabels( [] , visible=False )
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()

def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()


def describe_more( df ):
    var = [] ; l = [] ; t = []
    for x in df:
        var.append( x )
        l.append( len( pd.value_counts( df[ x ] ) ) )
        t.append( df[ x ].dtypes )
    levels = pd.DataFrame( { 'Variable' : var , 'Levels' : l , 'Datatype' : t } )
    levels.sort_values( by = 'Levels' , inplace = True )
    return levels

def plot_variable_importance( X , y ):
    tree = DecisionTreeClassifier( random_state = 99 )
    tree.fit( X , y )
    plot_model_var_imp( tree , X , y )
    
def plot_model_var_imp( model , X , y ):
    imp = pd.DataFrame( 
        model.feature_importances_  , 
        columns = [ 'Importance' ] , 
        index = X.columns 
    )
    imp = imp.sort_values( [ 'Importance' ] , ascending = True )
    imp[ : 10 ].plot( kind = 'barh' )
    print (model.score( X , y ))
    

# get titanic & test csv files as a DataFrame
train = pd.read_csv("C:/Users/Eugene/Desktop/Output/Titanic/train.csv")
test  = pd.read_csv("C:/Users/Eugene/Desktop/Output/Titanic/test.csv")
ids = test['PassengerId']
full = train.append( test , ignore_index = True )
titanic = full[ :891 ]
print ('Datasets:' , 'full:' , full.shape , 'titanic:' , titanic.shape)
#print(train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
#Turning variables to numeric ones and defining interactions for highly correlated variables
train['Sex'] = train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
test['Sex'] = test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
train = train.drop(['Ticket', 'Cabin','PassengerId'], axis=1)
test = test.drop(['Ticket', 'Cabin','PassengerId'], axis=1)
train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test['Title'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
freq_port = train.Embarked.dropna().mode()[0]
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Crew": 5, "Noble": 6}
port_mapping= {'S': 0, 'C': 1, 'Q': 2}
pclass_mapping= {1 : 'First', 2 : 'Second', 3 : 'Third'}
for dataset in [test,train]:
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col','Dr', 'Major', 'Rev'], 'Crew')
    dataset['Title'] = dataset['Title'].replace(['Don'], 'Mr')
    dataset['Title'] = dataset['Title'].replace(['Mlle','Ms','Dona'],'Miss')
    dataset['Title'] = dataset['Title'].replace(['Mme'], 'Mrs')
    dataset['Title'] = dataset['Title'].replace(['Lady','Countess','Sir', 'Jonkheer'], 'Noble')
    dataset['Pclass'] = dataset['Pclass'].map(pclass_mapping).astype(str)
#    dataset['Title'] = dataset['Title'].map(title_mapping).astype(int)
#    dataset['Title'] = dataset['Title'].fillna('None')
#    dataset=dataset.drop(['Name','SibSp','Parch'],axis=1)
train = train.drop(['Name'], axis=1)
test = test.drop(['Name'], axis=1)  
#train = train.drop(['Name','SibSp','Parch'], axis=1)
#test = test.drop(['Name','SibSp','Parch'], axis=1)    
train['Embarked'] = train['Embarked'].fillna(freq_port)   
test['Embarked'] = test['Embarked'].fillna(freq_port)
mid_fare= train.groupby('Embarked').mean()['Fare'][2]
test['Fare'] = test['Fare'].fillna(mid_fare)
train = pd.get_dummies(train,columns=['Pclass','Title','Embarked'])
test = pd.get_dummies(test,columns=['Pclass','Title','Embarked'])
test.insert(loc=13,column='Title_Noble', value=np.zeros(test.Sex.size))

#last step is predicting age variable basedon all data both for train and test
#In theory should be included in CrossValidation process in order to get right estimates

age_train = train[pd.notnull(train['Age'])]
age_fill = train[pd.isnull(train['Age'])].drop(['Age','Survived'],axis=1)
x_age = age_train.drop(['Age','Survived'],axis=1)
y_age = age_train['Age']
scaler=StandardScaler()
cv = ShuffleSplit(n_splits=5, test_size=0.2)
#x_age[['Fare']] = scaler.fit_transform(x_age[['Fare']])
"""C_range = np.logspace(3,6, 4)
gamma_range = np.logspace(-6,-2, 5)
SVR_param_grid = dict(gamma=gamma_range, C=C_range)

svr_rbf_age = SVR(cache_size=2000)
age_grid = GridSearchCV(svr_rbf_age, param_grid=SVR_param_grid, cv=cv, verbose=1, n_jobs=-1)
with parallel_backend('threading'):
    age_grid.fit(x_age, y_age)

print("The best parameters for method are %s with a score of %0.2f"
      % (age_grid.best_params_, age_grid.best_score_))"""

test_age = test[pd.isnull(test['Age'])].drop('Age',axis=1)
#svr_rbf = SVR(kernel='rbf', C=2.5e2, gamma=0.08,cache_size=3000)
svr_rbf = SVR(kernel='rbf', C=2.5e2, gamma=0.1,cache_size=2000)
svr_rbf.fit(x_age,y_age)
print('CV age score',cross_val_score(svr_rbf,x_age,y_age,cv=cv).mean())
#print(agelasso.score(x_age,y_age))
train_age_fill = svr_rbf.predict(age_fill)
test_age_fill = svr_rbf.predict(test_age)
print(train_age_fill)
train['Age'][pd.isnull(train['Age'])] = train_age_fill
test['Age'][pd.isnull(test['Age'])] = test_age_fill
print(train.info())
print(test.info())
scaler=StandardScaler()
#actuall fitting of survival
acc_score = make_scorer(accuracy_score)
cv = ShuffleSplit(n_splits=3, test_size=0.35)
x_norm = x_survival = train.drop('Survived',axis=1)
y_survival = train['Survived']
logreg = LogisticRegression()
logreg.fit(x_survival, y_survival)
print('logistic reg score: ',logreg.score(x_survival, y_survival))

svc_surv=SVC(gamma=0.1,C=1)
svc_surv.fit(x_survival, y_survival)
print('SVC score: ',svc_surv.score(x_survival, y_survival))

x_norm[['Fare','Age']] = scaler.fit_transform(x_norm[['Fare','Age']])
svc_nsurv = SVC(gamma=1e-1,C=1)
svc_nsurv.fit(x_norm, y_survival)
print('Normalized SVC score: ',svc_nsurv.score(x_survival, y_survival))

k=20
knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(x_survival, y_survival)
print(k,' nn score: ',knn.score(x_survival, y_survival))

gaussian = GaussianNB()
gaussian.fit(x_survival, y_survival)
print('Naive Bayes',gaussian.score(x_survival, y_survival))

bestdecision_tree = DecisionTreeClassifier(max_depth=10, max_features=12, max_leaf_nodes=9, min_samples_leaf=9)
decision_tree = DecisionTreeClassifier(max_depth=17, max_features=9, max_leaf_nodes=8, min_samples_leaf=8)
decision_tree.fit(x_survival, y_survival)
bestdecision_tree.fit(x_survival, y_survival)
print('DecisionTree ',decision_tree.score(x_survival, y_survival))
print('Best DecisionTree ',bestdecision_tree.score(x_survival, y_survival))

random_forest = RandomForestClassifier(n_estimators=100,max_depth=17, max_features=9, max_leaf_nodes=8, min_samples_leaf=8)
random_forest.fit(x_survival, y_survival)
print('Random Forest',random_forest.score(x_survival, y_survival))
method=[logreg,svc_surv,svc_nsurv,knn,gaussian,decision_tree,bestdecision_tree,random_forest]
scores=np.zeros(8)
scores[0]=cross_val_score(logreg,x_survival,y_survival,scoring=acc_score,cv=cv).mean()
scores[1]=cross_val_score(svc_surv,x_survival,y_survival,scoring=acc_score,cv=cv).mean()
scores[2]=cross_val_score(svc_surv,x_survival,y_survival,scoring=acc_score,cv=cv).mean()
scores[3]=cross_val_score(knn,x_survival,y_survival,scoring=acc_score,cv=cv).mean()
scores[4]=cross_val_score(gaussian,x_survival,y_survival,scoring=acc_score,cv=cv).mean()
scores[5]=cross_val_score(decision_tree,x_survival,y_survival,scoring=acc_score,cv=cv).mean()
scores[6]=cross_val_score(bestdecision_tree,x_survival,y_survival,scoring=acc_score,cv=cv).mean()
scores[7]=cross_val_score(random_forest,x_survival,y_survival,scoring=acc_score,cv=cv).mean()
print('Accuracy Logistic ', scores[0])
print('Accuracy SVC ',scores[1])
print('Accuracy NSVC ',scores[2])
print('Accuracy NN ',scores[3])
print('Accuracy Bayes ',scores[4])
print('Accuracy Tree ',scores[5])
print('Accuracy Best Tree ',scores[6])
print('Accuracy Best forest ',scores[7])
methodId=np.argmax(scores)
print('Best method' ,str(method[methodId]),' with score: ',scores[methodId])
predictions = method[methodId].predict(test)
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('titanic-predictions.csv', index = False)
output.head()
#x_survival[['Fare','Age']] = scaler.fit_transform(x_survival[['Fare','Age']])
C_range = np.logspace(-2,7, 10)
gamma_range = np.logspace(-3,3, 7)
md_range=np.arange(100); md_range[0] = 86
mxf_range=np.arange(1,18); mxf_range[0] = 12
lnodes_range=np.arange(1,10); lnodes_range[0] = 9
msleaf_range=np.arange(10); msleaf_range[0] = 8
forest_range=np.arange(1,21)
SVC_param_grid = dict(gamma=gamma_range, C=C_range)
DT_param_grid = dict( max_depth=md_range,max_features=mxf_range,max_leaf_nodes=lnodes_range,min_samples_leaf=msleaf_range)
RF_param_grid=dict(n_estimators=forest_range, max_depth=md_range)
cv = StratifiedShuffleSplit(n_splits=3, test_size=0.35)
r_f=DecisionTreeClassifier()
grid = GridSearchCV(r_f, param_grid=DT_param_grid, cv=cv, verbose=0, n_jobs=-1,scoring=acc_score)
with parallel_backend('threading'):
    grid.fit(x_survival, y_survival)

print("The best parameters for method are %s with a score of %0.4f"
      % (grid.best_params_, grid.best_score_))






