import pandas as pd
data_heart = pd.read_csv(r"C:/Users/User/Desktop/Hear_beat/heart_disease/data/heart.csv")
data_heart.head()
data_heart.columns.values
data_heart.dtypes
data_heart.shape
data_heart.describe()
data_heart.isnull()
print(len(data_heart) - data_heart.count())

import matplotlib.pyplot as plt
data_heart['HeartDisease'].value_counts().plot(kind='bar')
plt.show()

fig,ax = plt.subplots(figsize=(16,8))
ax.hist([data_heart[data_heart['HeartDisease']==1]['Age']])
ax.set_xlabel('Age')
ax.set_ylabel('Number of Heart Patients')
ax.set_xticks([30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80])
ax.set_title('Age Distribution')

import seaborn as sns
plt.figure(figsize=[16,8])
corr = sns.heatmap(data_heart.corr(), annot=True, cmap="RdYlGn")
plt.show()

#cols = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
#           'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
X = data_heart[cols]
Y = data_heart['HeartDisease']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state=0)

from sklearn import linear_model
from sklearn import metrics
lrm = linear_model.LogisticRegression(solver='lbfgs')
lrm.fit(X_train, Y_train)
probs = lrm.predict_proba(X_test)
print(probs)
predicted = lrm.predict(X_test)
print (predicted)

# SVM
# from sklearn import svm
# svm = svm.SVC()
# svm.fit(X_train, Y_train)

# predictions = svm.predict(X_test)

# print("Accuracy:",metrics.accuracy_score(Y_test, predictions))
# Accuracy: 0.6902173913043478

# DecisionTreeClassifier
# from sklearn.tree import DecisionTreeClassifier
# dt_heart = DecisionTreeClassifier(criterion='entropy',max_depth=5, min_samples_split=20, random_state=99)
# dt_heart.fit(X_train,Y_train)

# from sklearn.model_selection import KFold
# crossvalidation = KFold(n_splits=5, shuffle=True, random_state=1)

# from sklearn.model_selection import cross_val_score
# score = np.mean(cross_val_score(dt_heart, X_train, Y_train, scoring='accuracy', cv=crossvalidation, n_jobs=1))
# score
# 0.6403410679340229

# testY_predict = dt_heart.predict(X_test)
# print("Accuracy:", metrics.accuracy_score(Y_test, testY_predict))
# Accuracy: 0.75


# RandomForestClassifier
# from sklearn.ensemble import RandomForestClassifier
# clf=RandomForestClassifier(n_estimators=100)
# clf.fit(X_train,Y_train)

# y_predi = clf.predict(X_test)

# from sklearn import metrics
# # Model Accuracy, how often is the classifier correct?
# print("Accuracy:",metrics.accuracy_score(Y_test, y_predi))
# Accuracy: 0.657608695652174

# MLP Classifier    
# from sklearn.neural_network import MLPClassifier

# mlp_classifier  = MLPClassifier(random_state=123)
# mlp_classifier.fit(X_train, Y_train)
# Out[46]: MLPClassifier(random_state=123)

# Y_preds = mlp_classifier.predict(X_test)

# mlp_classifier.score(X_test, Y_test)
# Out[48]: 0.6902173913043478

print (metrics.accuracy_score(Y_test, predicted))
print(metrics.precision_score(Y_test, predicted))
print(metrics.recall_score(Y_test, predicted))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(linear_model.LogisticRegression(solver='lbfgs'), X, Y, 
scoring='accuracy', cv=5)
print (scores)

from sklearn.metrics import confusion_matrix
print("Confusion Matrix : \n\n" , confusion_matrix(Y_test, predicted))

import seaborn as sns
sns.heatmap(confusion_matrix(Y_test, predicted), annot=True)
plt.show()

import joblib 
joblib.dump(lrm, r'C:\Users\User\Desktop\Hear_beat\heart_disease\data/model_lrm.pkl')
print("Model dumped!")


model_columns = list(X.columns)
print(model_columns)
joblib.dump(model_columns, r'C:\Users\User\Desktop\Hear_beat\heart_disease\data/model_columns_lrm.pkl')
print("Models columns dumped!")

