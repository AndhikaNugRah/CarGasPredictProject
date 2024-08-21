import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn import linear_model

df=pd.read_csv('data_car.csv')
print(df)

#describes relationship
X= df[['Weight','Volume']]
Y=df['CO2']

#From the sklearn module we will use the LinearRegression() method to create a linear regression object.
regr_model=linear_model.LinearRegression()
regr_model.fit(X,Y)

#Testing model with weight 2300kg and volume 1300 cm cubic
TestCO2 = regr_model.predict([[2300, 1300]])

FinalResult=float(TestCO2)
print("The car with 2300 KG and 1300 cm cubic will release: ", FinalResult," CO2 Gases")

#Now we will test the The coefficient as a factor that describes the relationship with an variable.
Corr=regr_model.coef_
FinalCor=np.array([Corr])
print("If the weight increase by 1kg, the CO2 emission increases by :",Corr[0], "Gases")
print("If the Volume increase by 1kg, the CO2 emission increases by :",Corr[1], "Gases")

#What if the weight increases 1000
Test2CO2 = regr_model.predict([[3300, 1300]])

#calculation if increased 1000kg
IncreasedGas=0.0075509472703006895*1000
TestCal=FinalResult+IncreasedGas
Increased=TestCal-FinalResult

FinalResults=float(Test2CO2)
print("The car with 3300 KG and 1300 cm cubic will release: ", FinalResults," CO2 Gases")
print("If we calculate manual with correlation will results ", TestCal," CO2 Gases")
print("Before with 2200 kg we get:",FinalResult,"This mean we will get an increased gas about:",Increased, )