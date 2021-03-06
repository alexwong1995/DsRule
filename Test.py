import pandas as pd
from sklearn.linear_model import LogisticRegression
from DsRule_Base import DsRule

heart_disease=pd.read_csv("D:\\Kaggle\\heart-disease-uci\\heart.csv")
clf = LogisticRegression(C=1e5, solver='lbfgs',max_iter=1000)
X = heart_disease.drop(['target'],axis=1)
y = heart_disease['target']
clf.fit(X, y)
dsrule=DsRule(X, clf)
rule=dsrule.instance_explainer(X[0:1])
print(rule)
