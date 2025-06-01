import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Sun'iy ma'lumotlar
np.random.seed(42)
rooms = np.random.randint(1, 6, 100)
area = np.random.randint(30, 200, 100)
age = np.random.randint(0, 50, 100)
price = (rooms * 5000) + (area * 300) - (age * 1000) + np.random.randint(-10000, 10000, 100)

df = pd.DataFrame({
    'rooms': rooms,
    'area': area,
    'age': age,
    'price': price
})

# Model yaratish va o‘qitish
X = df[['rooms', 'area', 'age']]
y = df['price']
model = LinearRegression()
model.fit(X, y)

# Modelni saqlash
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ model.pkl fayli yaratildi!")
