import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as ply
import pydotplus as pdp
from sklearn.externals.six import StringIO
import pydotplus.graphviz as grapViz
from sklearn import tree

pd.options.mode.chained_assignment = None  # default='warn'
#Download the DataSet and replace the absolute path below
titanic_df=pd.read_csv('~/DataSets/titanic/titanicDataSet.csv')

# Convert the Sex classes to integer form
titanic_df["Sex"][titanic_df["Sex"]=='male']=1
titanic_df["Sex"][titanic_df["Sex"]=='female']=0

# Convert the Embarked classes to integer form
titanic_df["Embarked"][titanic_df["Embarked"] == "S"] = 0
titanic_df["Embarked"][titanic_df["Embarked"] == "C"] = 1
titanic_df["Embarked"][titanic_df["Embarked"] == "Q"] = 2

#Fill in the Nan
titanic_df.Fare[152] = titanic_df["Fare"].median()

#Fill Nan in Age with Median
titanic_df["Age"]=titanic_df["Age"].fillna(titanic_df["Age"].median())

#Fill Nan in Embarked with MEdian
titanic_df["Embarked"]=titanic_df["Embarked"].fillna(titanic_df["Embarked"].median())

#Adding new Feature Family Size
titanic_df["FamilySize"]=titanic_df["SibSp"]+titanic_df["Parch"]+1

trainData=titanic_df[["Pclass","Sex","Age","Fare", "SibSp", "Parch","Embarked","FamilySize"]].values
trainTargets=titanic_df[["Survived"]].values.ravel()

tianicTree=DecisionTreeClassifier(max_depth =7, min_samples_split = 15, random_state = 17)
tianicTree.fit(trainData,trainTargets)

scores = cross_val_score(tianicTree, trainData, trainTargets, cv=10)
print 'Tree Targets ',tianicTree.score(trainData,trainTargets)
print (tianicTree.feature_importances_)
print 'Scores', scores.mean()
