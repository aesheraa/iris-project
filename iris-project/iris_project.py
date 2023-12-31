

# yapay zeka kullanarak iris �i�e�i t�r�n�  analiz etme projesi


# # PCA -  Principal Component Analysis
#Iris veri seti 3 Iris bitki t�r�ne (Iris setosa, Iris virginica ve Iris versicolor) ait, her bir t�rden 50 �rnek olmak �zere toplam 150 �rnek sayisina sahip bir veri setidir. 
#Her bir �rnek i�in 4 �zellik tanimlanmistir: ta� yaprak uzunlugu, ta� yaprak genisligi, �anak yaprak genisligi, �anak yaprak uzunlu�u('sepal length','sepal width','petal length','petal width'). 
# Veri setimizde, her bir bitki �rnegi ayri bir g�zlemi (�rnegi) ifade ederken; bitki t�r ismi bagimli(dependent) degisken, 
#bitkilerin �l��len 4 temel �zelligi ise bagimsiz(independent) degiskenleri ifade eder.



#PCA ile �ok boyutlu veri setlerini, anlam�n� kaybetmeden daha az boyutlu hale getirmeyi ve  
#�ok fazla boyutlu veri setlerini kolay g�rselle�tirebilmek i�in ve insan g�z�yle 2 veya 3 boyutlu vereri rahat�a g�r�p anlasilmasini ama�l�yorum.
# �rne�in Verileri s�k��t�rmak ve b�y�k boyuttaki dijital foto�raf dosyalar�n� daha k���k formatlara s�k��t�rmak i�in.



#Baz� veri setleri �ok feature i�erdi�inde bunlar�n depolanmas� ve analiz edilmesini PCA i�lemi kolayla�t�r�r.
#PCA yapma i�lemi matematiksel bir i�lemdir. Eigenvekt�rler ��kar�l�r ve PCA 1. Boyut ve PCA 2. Boyut... N. Boyuta kadar indirgenir.
#N = d���r�lmek istenen boyut say�s�d�r.
#Bu i�lemler yap�l�rken varyans  %90 dan fazla oranda korunmaya �al���l�r. Yani minimum kay�p ger�ekle�ir.


import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

url = "pca_iris.data"
# datasetimizi Pandas DataFrame i�ine y�kl�yoruz..
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])

df


features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# feature'lar� x olarak ay�ral�m:
x = df[features]

# target'i y olarak ay�ral�m:
y = df[['target']]


# De�erleri Scale etmemiz gerekiyor. ��nk� her bir feature �ok farkl� boyutlarda ve bunlar�n yapay zeka taraf�ndan e�it a��rl�klarda dengelenmesi gerekiyor. 
# Bu ama�la standart scaler  kullanarak t�m verileri mean = 0 and variance = 1 olacak �ekilde de�i�tiriyoruz.


# Standardizing the features
x = StandardScaler().fit_transform(x)



# Bakal�m scale etmi� mi?
x


#�ris �i�e�inin 4 boyutlu veri setini 2 boyuta indirgiyorum. 
#Ben PCA ile bunu 2 boyuta indirgeyecegim ancak sonucunda elde edece�imiz 2 boyut herhangi bir anlam ifade etmeyen ba�l�klara sahip olacak. 
#Yani 4 feature dan 2 tanesini basit bir �ekilde atmak de�il yapt���m.


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])


principalDf


# �imdi en son target s�tunumuzu da PCA dataframe'imizin sonuna ekleyelim:



final_dataframe = pd.concat([principalDf, df[['target']]], axis = 1)


final_dataframe.head()



# Son olarak da final dataframe'imizi g�rselle�tirip bakal�m:

# Basit bir �izim yapal�m:


dfsetosa= final_dataframe[df.target=='Iris-setosa']
dfvirginica = final_dataframe[df.target=='Iris-virginica']
dfversicolor = final_dataframe[df.target=='Iris-versicolor']
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')

plt.scatter(dfsetosa['principal component 1'], dfsetosa['principal component 2'],color='green')
plt.scatter(dfvirginica['principal component 1'], dfvirginica['principal component 2'],color='red')
plt.scatter(dfversicolor['principal component 1'], dfversicolor['principal component 2'],color='blue')


#Daha profesyonel bir plotting yapal�m:
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['g', 'b', 'r']

plt.xlabel('principal component 1')
plt.ylabel('principal component 2')

for target, col in zip(targets,colors):
    dftemp = final_dataframe[df.target==target]
    plt.scatter(dftemp['principal component 1'], dftemp['principal component 2'], color=col)



pca.explained_variance_ratio_


pca.explained_variance_ratio_.sum()
