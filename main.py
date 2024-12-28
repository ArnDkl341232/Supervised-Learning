import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


#   Data

data = {
    "Days": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100],
    "Candies":[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200]
}

df = pd.DataFrame(data)
print(df)

#   Data visualisation

plt.scatter(df["Days"],df["Candies"],color = "blue")
plt.title("Graph dependence")
plt.xlabel("Days")
plt.ylabel("Candies")
plt.show()

#   Data separation

X = df[["Days"]]
Y = df["Candies"]

#   Model creation and teaching
model = LinearRegression()
model.fit(X,Y)

print("Weight (W):", model.coef_)
print("bias (b):" , model.intercept_)

#  Prediction

Y_pred = model.predict(X)


#   Graph with regretion line

plt.scatter(X, Y,color = "blue")
plt.plot(X, Y_pred,color = "red")
plt.title("Regretion line")
plt.xlabel("Days")
plt.ylabel("Candies")
plt.show()


#   Prediction

new_value = [[10]]
predicted_candies = model.predict(new_value)
print(f"Prediction for {new_value[0][0]}:" , predicted_candies[[0]])















