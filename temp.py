from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
#EQUIPE
#BÁRBARA SANTOS FREITAS
#FRANCELINO GIORDANI
#MAYARA ASSIS NASCIMENTO

iris = load_iris()
iris.keys()

n=iris['target_names'] #tipos de iris
#print(n)
n2=iris['feature_names'] #cada uma das caracteristicas 
#print(n2)
print(iris['data'].shape)
print("\n")
iris['data'][:150]
#print(iris['data'][:150] )
print("Dados classificados de 0 a 2\n")
iris['target'] #target classifica as flores de 0 a 2
print(iris['target'])
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'],test_size=0.2, random_state = 0)
#divide os dados para teste e para treinamento
print ("\nDados para treino: ")
print(X_train.shape)
print("Dados para teste: ")
print(X_test.shape)
print("\n")
print("        valores de teste X \n ")
print(X_test)
print("\n")
print("        valores de teste y \n")
print(y_test)
print("\n")
knn = KNeighborsClassifier(n_neighbors = 1)

knn.fit(X_train, y_train)


#testando modelo
X_new = np.array([
 [5.8,2.8,5.1,2.4]
,[6.,2.2,4.,1.,]
,[5.5,4.2,1.4,0.2]
,[7.3,2.9,6.3,1.8]
,[5.,3.4,1.5,0.2]
,[6.3,3.3,6.,2.5]
,[5.,3.5,1.3,0.3]
,[6.7,3.1,4.7,1.5]
,[6.8,2.8,4.8,1.4]
,[6.1,2.8,4.,1.3]
,[6.1,2.6,5.6,1.4]
,[6.4,3.2,4.5,1.5]
,[6.1,2.8,4.7,1.2]
,[6.5,2.8,4.6,1.5]
,[6.1,2.9,4.7,1.4]
,[4.9,3.1,1.5,0.1]
,[6.,2.9,4.5,1.5]
,[5.5,2.6,4.4,1.2]
,[4.8,3.,1.4,0.3]
,[5.4,3.9,1.3,0.4]
,[5.6,2.8,4.9,2.,]
,[5.6,3.,4.5,1.5]
,[4.8,3.4,1.9,0.2]
,[4.4,2.9,1.4,0.2]
,[6.2,2.8,4.8,1.8]
,[4.6,3.6,1.,0.2]
,[5.1,3.8,1.9,0.4]
,[6.2,2.9,4.3,1.3]
,[5.,2.3,3.3,1.,]
,[5.,3.4,1.6,0.4]
 
])

X_new.shape

prediction = knn.predict(X_new)
print("Prediction \n")
print(prediction)
print("\n")
nome=iris['target_names'][prediction]
#print(nome)
print("\n")
acertos=0
for i in prediction:
  if prediction[i] == 0:
      if prediction[i]==y_test[i]:
         acertos=acertos+1
         print("setosa")
  if prediction[i]==1:
       if prediction[i]==y_test[i]:
         acertos=acertos+1
         print("versicolor")
  if prediction[i]==2:
       if prediction[i]==y_test[i]:
         acertos=acertos+1
         print("virginica")
         


print("\n\n")
print("acertos")
print(acertos)
    

#Accuracy do modelo
print("Precisao: ")
print(knn.score(X_test, y_test))

