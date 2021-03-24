import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix

# missing value

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
# x feature
# y label

x = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)



clf = neighbors.KNeighborsClassifier(n_neighbors=5)
clf.fit(x_train, y_train)
p = clf.predict(x_test)
result=list()
for i in p:
    if i==2:
        result.append('Begin')
    else:
        result.append('Malignant')
print(result)        
from sklearn import metrics
print("Accuracy:", metrics.accuracy_score(y_test, p))
df.hist(figsize=(10,10))
plt.show()
scatter_matrix(df, figsize=(10,10))
plt.show()
print('---------------------------------------------------------','\n')




print('-------------------------Test User-----------------------')
example_measure = np.array([[4, 2, 5, 1, 4, 2, 4, 2, 1], [4, 2, 1, 1, 1, 2, 3, 2, 1]])
b = example_measure
example_measure = example_measure.reshape(len(example_measure), -1)
predection = clf.predict(example_measure)
result=list()
for i in predection:
    if i==2:
        result.append('Begin')
    else:
        result.append('Malignant')
print(result)
print('---------------------------------------------------------','\n')




print('-------------artificial neural network-------------------','\n')
# this is neural network
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(9), max_iter=500)
mlp.fit(x_train, y_train)
y_pred = mlp.predict(x_test)
result=list()
for i in p:
    if i==2:
        result.append('Begin')
    else:
        result.append('Malignant')
print(result)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print('\n','confusion_matrix')
print(confusion_matrix(y_test, y_pred))
print('\n','classification_report')
print(classification_report(y_test, y_pred))
print('---------------------------------------------------------','\n')



print('-------------this is decision tree-------------------','\n')
# this is decision tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=10)
dt.fit(x_train, y_train)
dt_predict = dt.predict(x_test)
result=list()
for i in p:
    if i==2:
        result.append('Begin')
    else:
        result.append('Malignant')
print(result)
dt_conf_matrix = confusion_matrix(y_test, dt_predict)
dt_accuracy_score = accuracy_score(y_test, dt_predict)
print('\n','confusion_matrix')
print(dt_conf_matrix)
print('\n','accuracy')
print(dt_accuracy_score)
print('---------------------------------------------------------','\n')




print('-------------this is KNN with using PCA-------------------','\n')
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
# print(x)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x_new = pca.fit_transform(x)
x_new_train, x_new_test, y_new_train, y_new_test = train_test_split(x_new, y, test_size=0.2)
# apply KNN with using PCA
claf = neighbors.KNeighborsClassifier(n_neighbors=5)
claf.fit(x_new_train, y_new_train)
predict_x_new = claf.predict(x_new_test)

conf_matrix = confusion_matrix(y_new_test, predict_x_new)
accuracy_score = accuracy_score(y_new_test, predict_x_new)
print('this is KNN with using PCA')
print(conf_matrix)
print(accuracy_score)

colors = ['r', '', 'b']
test_colors = ['y', '', 'g']

plt.figure(figsize=(10, 9))
print(len(x_new))

for i in range(len(x_new)):
    plt.scatter(x_new[i][0], x_new[i][1], c=colors[y[i] - 2], s=5)

for i in range(len(x_new_test)):
    plt.scatter(x_new_test[i][0], x_new_test[i][1], c=test_colors[predict_x_new[i] - 2], marker='+')

# print(clf.score(example_measure,predection))

