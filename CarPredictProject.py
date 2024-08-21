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
print("Before with 2300 kg we get:",FinalResult,"This mean we will get an increased gas about:",Increased, )

def predict_co2():
    while True:
        try:
            X1 = int(input("Input your Weight (in Kilogram): "))
            X2 = int(input("Input your Volume (in CM Cubic): "))
            break
        except ValueError:
            print("Invalid input. Please enter an integer value.")

    TestCO2 = regr_model.predict([[X1, X2]])
    FinalResult = float(TestCO2)
    print(f"The car with {X1} KG and {X2} cm cubic will release: {FinalResult} CO2 Gases")

    cont = input("Do you want to continue? (yes/no): ")
    if cont.lower() == "yes":
        predict_co2()  # recursive call to go back to the beginning
    else:
        print("Goodbye, thanks for using our service!  - Dhika")

predict_co2()  # initial call to start the loop
