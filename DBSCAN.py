

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

#Importing dataset
dataset = pd.read_csv("Mall_Customers.csv")
X=dataset.iloc[:,[3,4]].values

#Applying DBSCAN algorithm
from sklearn.cluster import DBSCAN
dbscan=DBSCAN(eps=3,min_samples=6)

#Fitting the model
model=dbscan.fit(X)
cluster=model.labels_ #it fing the no. of clusters and -1 value indicate the noise or outlier point.
len(set(cluster))

#Identifying the points which mahes up our core points
sample_cores = np.zeros_like(labels,dtype=bool)
sample_cores[dbscan.core_sample_indices_]=True

#calculating the number of clusters
n_clusters=len(set(labels))-(1 if -1 in labels else 0)
print(metrics.silhouette_score(X,labels))

#Plotting the clusters
df = DataFrame(dict(x=X[:,0],y=X[:,1],label=cluster))
colors = {-1:'red',0:'blue',1:'yellow',2:'green',3:'magenta'}
fig,ax=plt.subplots(figsize=(8,8))
grouped=df.groupby('label')
for key,group in grouped:
	group.plot(ax=ax,kind='scatter',x='x',y='y',label=key,color=colors[key])
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.show()

show_clusters(X,cluster)
