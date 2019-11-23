import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import KMeans
import matplotlib.cm as cm
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from sklearn.metrics import silhouette_samples 
from sklearn.metrics import silhouette_score  #avg of avgs


def calc_threshold(column_name, multiplier):
	Q1 = df_num[column_name].quantile(0.25)
	Q3 = df_num[column_name].quantile(0.75)
	IQR = Q3 - Q1
	return IQR * multiplier

def get_outliers_i(column, multiplier):
	if multiplier == 0:
		return []
	th_pos = calc_threshold(column, multi) + df_num[column].mean()
	th_neg = df_num[column].mean() - calc_threshold(column, multi)
	outliers_i = df_num[(df_num[column] >= th_pos) | (df_num[column] <= th_neg)].index.values
	return outliers_i

def create_silgraph(df, labels):
	sample_silhouette_values = silhouette_samples(df, labels )
	
	n_clusters = len(labels.unique())
	y_lower = 100
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	
	for i in range(n_clusters):
	    ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
	    ith_cluster_silhouette_values.sort()
	    size_cluster_i=ith_cluster_silhouette_values. shape[0]
	    y_upper = y_lower + size_cluster_i
	    y_upper = y_lower + size_cluster_i
	    color = cm.nipy_spectral(float(i) / n_clusters)
	    ax1.fill_betweenx(np.arange(y_lower, y_upper),0, ith_cluster_silhouette_values,facecolor=color, edgecolor=color, alpha=0.7)
	
	    # Label the silhouette plots with their cluster numbers at the middle
	    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
	
	    # Compute the new y_lower for next plot
	    y_lower = y_upper + 10  # 10 for the 0 samples
