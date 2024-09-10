import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

df = pd.read_csv("train.csv")

df = df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']]

df = df.dropna()

x = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df['SalePrice']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = LinearRegression()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test,y_pred)
print()
print(f"Mean Squared Error between predicted and actual price is : {mse:.2f}")

print()
print()

r2 = r2_score(y_test,y_pred)
print(f"R-Squared Value : {r2:.4f}")
print()

with open('HousePricePredictor','wb') as file:
    pickle.dump(model,file)

