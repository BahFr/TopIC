#1.Dados da base Diabetes
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
#itens da base
diabetes.keys()
print(diabetes.DESCR)
#pandas para manipular os dados em tabelas
import pandas
tabela = pandas.DataFrame(diabetes.data)
tabela.columns = diabetes.feature_names
print(tabela.head(10))
tabela['YDiabete'] = diabetes.target
tabela.head(10)
