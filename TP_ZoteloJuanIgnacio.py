import pandas as pd 
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics 

######################################  SEGUROS O DATA ######################################
dataset =pd.read_csv('C:/Users/elkuani/Downloads')  
print(dataset)

################### verifico la cantidad de filas y columnas ######################################
print("Numero de Filas & Columnas:", dataset.shape)

######################################  aca corroboro de no tener ningun valor Na N##############################

print("\nTengo valores NaN: ", dataset.isnull().values.any())
print(dataset.describe())

################## Aca separo las variables a comparar #############
# X=dataset.iloc[:,:-1].values
# Y=dataset.iloc[:,-1].values
#                OTRA MANERA DE HACER LA SEPARACION DE DATOS
X = dataset[['age','sex','bmi','children','smoker','region' ]].values
Y = dataset["charges"].values

print("\n X :\n ", X)
print("\n Y :\n ", Y)


####################################### MODIFICACIONES DE LAS COLUMNAS ##################################

#                EDAD 

labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
#                MODIFICO EL SEXO 
labelencoder_X=LabelEncoder()
X[:,1]=labelencoder_X.fit_transform(X[:,1])
#                MODIFICO SMOKER  
labelencoder_X=LabelEncoder()
X[:,4]=labelencoder_X.fit_transform(X[:,4])
#                MODIFICO LA CIUDAD  
labelencoder_X=LabelEncoder()
X[:,5]=labelencoder_X.fit_transform(X[:,5])

#                CORROBORO LAS MODIFICACIONES
print("\n X: \n  ",X)
print("\n Y: \n",Y)

################################################# GRAFICO ############################################################################                 
plt.figure(figsize=(8,5))
plt.title('Densidad') 
plt.tight_layout()                                           # exploto con dataset.plot(kind='bar', colormap='Greens')
seabornInstance.distplot(dataset['charges'])
plt.show()

##################################### CONJUNTO DE ENTRENAMIENTO 0.80 % Y CONJUNTO DE TEST 0.20 % ######################################
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

####################################  REGRESSION LINEAL ###################################################### 
regressor = LinearRegression() 
regressor.fit(X_train, y_train)

####################################  CREO UN DATAFRAME CON LOS INDICES          ####################################
dataframe = dataset.drop(['charges'], axis=1)
dataframe = dataframe.T
dataframe = dataframe.index
print('\n Indices: ', dataframe)


####################################  COEFICIENTE ####################################  

coeficiente_dataframe = pd.DataFrame(regressor.coef_, dataframe, columns=['Coeficiente'])
print('\n Coeficiente: \n\n', coeficiente_dataframe)

#################################### PREDICCION DE LOS TEST ####################################
y_pred = regressor.predict(X_test)

#################################### COMPARO LOS VALORES CON LA PREDICCION ####################################

dataframe = pd.DataFrame({'Actual': y_test, 'Prediccion': y_pred})
dataframe1 = dataframe.head(15)
print('\n Data Frame con la comparacion :\n\n', dataframe1)


####################################       GRAFICO COMPARACIONS       ####################################
# Se pueden obtener otras figuras modificando " KIND " : [ bar, box  , hist ,kde ] , modificando STACKED = true se pisan los colores  

dataframe1.plot(kind='bar', stacked=False ,figsize=(8,5))
plt.title('Prediccion')  
plt.grid(which='major', linestyle='-', linewidth='1', color='black')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='white')  
plt.show()

dataframe1.plot(kind='area', stacked=False ,figsize=(8,5))   
plt.title('Prediccion con area') 
plt.grid(which='major', linestyle='-', linewidth='1', color='black')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='white')  
plt.show()

####################################      ERRORES           ###########################################

print('\n Error Absoluto Medio:', metrics.mean_absolute_error(y_test, y_pred)) 
print('\n Error Cuadrático Medio:', metrics.mean_squared_error(y_test, y_pred)) 
print('\n Raíz del Error Cuadrático Medio :', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("\n")

##################################        Conclusion    ###############################################
# En el primer grafico se puede ver que las mayores coincidencias esta en el rango de los valores 0 y 15000 aproximadamente. El segundo grafico esta
# realizado con las predicciones y observando los resultados obtenidos de los errores ( Error Absoluto Medio , Error Cuadrático Medio , Raíz del Error Cuadrático Medio) 
# no seria el esperado , ya que supera el 10 por ciento.

################################## BIBLIOGRAFIA ###################################
#                                  PLOTING : utilizado en el tercer grafico 
# PAGINAS :
#   https://pandas.pydata.org/pandas-docs/version/0.15.0/visualization.html
#   https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html#visualization-box




