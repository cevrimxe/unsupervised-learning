# Gerekli kütüphaneleri import edelim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = {
    'Age': [25, 45, 35, 50, 23, 34, 30, 40, 60, 29, 70, 55, 41, 35, 50, 22, 44, 33, 38, 47,
            28, 60, 59, 35, 48, 24, 32, 53, 30, 29, 44, 26, 34, 41, 31, 52, 63, 29, 43, 49,],
    'Annual Income (k$)': [15, 60, 45, 80, 25, 40, 30, 60, 100, 35, 20, 70, 85, 60, 55, 20, 55, 40, 75, 100,
                           60, 95, 50, 45, 85, 25, 40, 70, 65, 55, 50, 30, 45, 70, 40, 55,  65, 90, 35, 60],
    'Spending Score (1-100)': [39, 81, 6, 77, 40, 76, 6, 94, 3, 72, 90, 45, 63, 54, 45, 72, 58, 35,
                               81, 56, 50, 28, 39, 57, 48, 65, 50, 55, 46, 41, 63, 30, 28, 72, 54, 61,
                               45, 32, 75, 68],
}

# Pandas DataFrame'e dönüştür
df = pd.DataFrame(data)

# Veriyi ölçeklendir
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

# Elbow method ile küme sayısını belirleme
inertia = []
for i in range(1, 11):  # 1'den 10'a kadar kümeleri deneyelim
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Sonuçları görselleştirme
plt.plot(range(1, 11), inertia)
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# K-means modelini oluştur ve kümeleri bul
kmeans = KMeans(n_clusters=4)  # Küme sayısını 4 olarak belirledik
kmeans.fit(scaled_data)

# Kümeleme sonuçlarını dataframe'e ekleyelim
df['Cluster'] = kmeans.labels_

# Kümeleme sonuçlarını görselleştirelim
plt.figure(figsize=(8, 6))
plt.scatter(df['Age'], df['Annual Income (k$)'], c=df['Cluster'], cmap='viridis')
plt.title('Customer Segmentation using K-means')
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.show()

# Küme özelliklerini inceleyelim
print(df.groupby('Cluster').mean())
