#first feature
weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']

#second feature
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

#label or target 
play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

#encode Overcast:0, Rainy:1, and Sunny:2.
#hot=1 mild=2 cool=0
#no=0 yea=1

from sklearn import preprocessing

le=preprocessing.LabelEncoder()

weather_encoded=le.fit_transform(weather)
temp_encoded=le.fit_transform(temp)
label=le.fit_transform(play)

print(weather_encoded)
print(temp_encoded)
print(label)

#combinig two featurea
features=(zip(weather_encoded,temp_encoded))

#build KNN classifier model

from sklearn.neighbors import KNeighborsClassifier
try:
    model = KNeighborsClassifier(n_neighbors=3)
    x,y=model.fit(features,label)
except ValueError:
    print("not accept")
    
   