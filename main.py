import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Генеруємо дані
np.random.seed(24)
hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
energy_usage = 10 * hours + 15 + np.random.normal(0, 5, len(hours))  # Додаємо шум

# Вхідні та вихідні дані
X = hours.reshape(-1, 1)  # Перетворюємо в 2D-масив
Y = energy_usage  # Вихідні дані

#   Model creation and teaching
model = LinearRegression()
model.fit(X,Y)

print("Weight (W):", model.coef_)
print("bias (b):" , model.intercept_)

#  Prediction

Y_pred = model.predict(X)


#   Graph with regretion line

plt.scatter(X, Y,color = "blue", label="Real data")
plt.plot(X, Y_pred,color = "red", label="Regretion line")
plt.title("Regretion line:energy spends from work hours")
plt.xlabel("Work time")
plt.ylabel("Energy spends(Wat)")
plt.show()


#   Prediction

new_value = [[1000]]
predicted_candies = model.predict(new_value)
print(f"Prediction for {new_value[0][0]}:" , predicted_candies[[0]])















