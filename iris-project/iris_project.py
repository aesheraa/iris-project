

# yapay zeka kullanarak iris çiçeði türünü  analiz etme projesi


# # PCA -  Principal Component Analysis
#Iris veri seti 3 Iris bitki türüne (Iris setosa, Iris virginica ve Iris versicolor) ait, her bir türden 50 örnek olmak üzere toplam 150 örnek sayisina sahip bir veri setidir. 
#Her bir örnek için 4 özellik tanimlanmistir: taç yaprak uzunlugu, taç yaprak genisligi, çanak yaprak genisligi, çanak yaprak uzunluðu('sepal length','sepal width','petal length','petal width'). 
# Veri setimizde, her bir bitki örnegi ayri bir gözlemi (örnegi) ifade ederken; bitki tür ismi bagimli(dependent) degisken, 
#bitkilerin ölçülen 4 temel özelligi ise bagimsiz(independent) degiskenleri ifade eder.



#PCA ile çok boyutlu veri setlerini, anlamýný kaybetmeden daha az boyutlu hale getirmeyi ve  
#çok fazla boyutlu veri setlerini kolay görselleþtirebilmek için ve insan gözüyle 2 veya 3 boyutlu vereri rahatça görüp anlasilmasini amaçlýyorum.
# örneðin Verileri sýkýþtýrmak ve büyük boyuttaki dijital fotoðraf dosyalarýný daha küçük formatlara sýkýþtýrmak için.



#Bazý veri setleri çok feature içerdiðinde bunlarýn depolanmasý ve analiz edilmesini PCA iþlemi kolaylaþtýrýr.
#PCA yapma iþlemi matematiksel bir iþlemdir. Eigenvektörler çýkarýlýr ve PCA 1. Boyut ve PCA 2. Boyut... N. Boyuta kadar indirgenir.
#N = düþürülmek istenen boyut sayýsýdýr.
#Bu iþlemler yapýlýrken varyans  %90 dan fazla oranda korunmaya çalýþýlýr. Yani minimum kayýp gerçekleþir.


import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

url = "pca_iris.data"
# datasetimizi Pandas DataFrame içine yüklüyoruz..
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])

df


features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# feature'larý x olarak ayýralým:
x = df[features]

# target'i y olarak ayýralým:
y = df[['target']]


# Deðerleri Scale etmemiz gerekiyor. Çünkü her bir feature çok farklý boyutlarda ve bunlarýn yapay zeka tarafýndan eþit aðýrlýklarda dengelenmesi gerekiyor. 
# Bu amaçla standart scaler  kullanarak tüm verileri mean = 0 and variance = 1 olacak þekilde deðiþtiriyoruz.


# Standardizing the features
x = StandardScaler().fit_transform(x)



# Bakalým scale etmiþ mi?
x


#Ýris çiçeðinin 4 boyutlu veri setini 2 boyuta indirgiyorum. 
#Ben PCA ile bunu 2 boyuta indirgeyecegim ancak sonucunda elde edeceðimiz 2 boyut herhangi bir anlam ifade etmeyen baþlýklara sahip olacak. 
#Yani 4 feature dan 2 tanesini basit bir þekilde atmak deðil yaptýðým.


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])


principalDf


# Þimdi en son target sütunumuzu da PCA dataframe'imizin sonuna ekleyelim:



final_dataframe = pd.concat([principalDf, df[['target']]], axis = 1)


final_dataframe.head()



# Son olarak da final dataframe'imizi görselleþtirip bakalým:

# Basit bir çizim yapalým:


dfsetosa= final_dataframe[df.target=='Iris-setosa']
dfvirginica = final_dataframe[df.target=='Iris-virginica']
dfversicolor = final_dataframe[df.target=='Iris-versicolor']
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')

plt.scatter(dfsetosa['principal component 1'], dfsetosa['principal component 2'],color='green')
plt.scatter(dfvirginica['principal component 1'], dfvirginica['principal component 2'],color='red')
plt.scatter(dfversicolor['principal component 1'], dfversicolor['principal component 2'],color='blue')


#Daha profesyonel bir plotting yapalým:
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['g', 'b', 'r']

plt.xlabel('principal component 1')
plt.ylabel('principal component 2')

for target, col in zip(targets,colors):
    dftemp = final_dataframe[df.target==target]
    plt.scatter(dftemp['principal component 1'], dftemp['principal component 2'], color=col)



pca.explained_variance_ratio_


pca.explained_variance_ratio_.sum()
