import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Örnek veri (normal ve anormal harcamalar)
data = np.array([[20000], [15000], [12000], [8000], [10000], [50000], [20000], [25000], [30000], [30]])

# Anomali tespiti modelini oluştur
model = IsolationForest(contamination=0.2)  # 0.2 anomali oranı
model.fit(data)

# Tahmin yap
pred = model.predict(data)
print(pred)  # -1 anomali, 1 normal

# Görselleştir
plt.scatter(range(len(data)), data, c=pred, cmap='coolwarm')
plt.xlabel('Index')
plt.ylabel('Spending')
plt.title('Anomaly Detection')
plt.show()
