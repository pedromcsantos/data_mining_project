import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from sklearn.metrics import silhouette_score  #avg of avgs
from helperFunctions import create_silgraph, get_outliers_i, create_elbowgraph
import plotly.offline as pyo
import plotly.figure_factory as ff
import plotly.graph_objects as go

	
	
df = pd.read_csv("data/A2Z Insurance.csv")

# Preprocessing
df = df.set_index("Customer Identity")
newnames = ["first_policy","birth_year","educ","salary_monthly","location","has_children","mon_value","claims_rate","premium_motor","premium_household","premium_health","premium_life","premium_work_comp"]
df.rename(columns=dict(zip(df.columns.values, newnames)), inplace = True)
df["salary_monthly"] = df["salary_monthly"]*12 #multiple salary by 12 to have everything in the same unit (year)
df.rename(columns={"salary_monthly":"salary_year"}, inplace = True)
df = df.drop("birth_year", axis=1) #Drop birth_year for clustering; consider it for interpretation
df = df[df["first_policy"]<50000] #Drop one case where first_policy year <50000 


# Plot distribution for all values
num_col = ['first_policy','salary_year', 'mon_value','claims_rate','premium_motor','premium_household','premium_health','premium_life','premium_work_comp']
bin_size = [0, 10000, 300, 0.4, 50,50,50,50,50]
for i in range(0,len(num_col)):
	x = df.loc[:,num_col[i]]
	x = x.dropna()
	hist_data = [x]
	group_labels = ['distplot']

	fig = ff.create_distplot(hist_data, group_labels, bin_size=bin_size[i])
	fig.update_layout(title_text='Distplot for ' + num_col[i])
	fig.write_image("images/distplots/with_outlier/" + num_col[i] + "_distplot.png")

# Plot distribution without extreme outliers
df = df[df["salary_year"] < 200000]
df = df[df["mon_value"] > -50000 ]
df = df[df["claims_rate"] < 20]
df = df[df["premium_motor"] < 2000]
df = df[df["premium_household"] < 3000]
df = df[df["premium_health"] < 5000]
df = df[df["premium_life"] < 400]
df = df[df["premium_work_comp"] < 500]

num_col = ['first_policy','salary_year', 'mon_value','claims_rate','premium_motor','premium_household','premium_health','premium_life','premium_work_comp']
bin_size = [0, 10000, 300, 0.4, 50,50,50,50,50]
for i in range(0,len(num_col)):
	x = df.loc[:,num_col[i]]
	x = x.dropna()
	hist_data = [x]
	group_labels = ['distplot']

	fig = ff.create_distplot(hist_data, group_labels, bin_size=bin_size[i])
	fig.update_layout(title_text='Distplot for ' + num_col[i])
	fig.write_image("images/distplots/without_outlier/" + num_col[i] + "_distplot.png")


# After plotting the features we decided to test the following outlier variations:
# Customer related features


cluster_centroids = []
variations = [
		{'salary_year': 0,'mon_value': 0,'claims_rate': 0,'premium_motor': 0,'premium_household': 0,'premium_health': 0,'premium_life': 0,'premium_work_comp': 0}, # include all, good sil score but centroids too skewed by outliers
		{'salary_year': 200000,'mon_value': -50000,'claims_rate': 20,'premium_motor': 2000,'premium_household': 3000,'premium_health': 5000,'premium_life': 400,'premium_work_comp': 500}, # exclude extreme outlier
		{'salary_year': 200000,'mon_value': -1000,'claims_rate': 8,'premium_motor': 600,'premium_household': 1600,'premium_health': 400,'premium_life': 300,'premium_work_comp': 300},
		{'salary_year': 200000,'mon_value': -1000,'claims_rate': 3,'premium_motor': 600,'premium_household': 1600,'premium_health': 400,'premium_life': 300,'premium_work_comp': 300},
		]

for v in variations:
	df = pd.read_csv("data/A2Z Insurance.csv")

	# Preprocessing
	df = df.set_index("Customer Identity")
	newnames = ["first_policy","birth_year","educ","salary_monthly","location","has_children","mon_value","claims_rate","premium_motor","premium_household","premium_health","premium_life","premium_work_comp"]
	df.rename(columns=dict(zip(df.columns.values, newnames)), inplace = True)
	df["salary_monthly"] = df["salary_monthly"]*12 #multiple salary by 12 to have everything in the same unit (year)
	df.rename(columns={"salary_monthly":"salary_year"}, inplace = True)
	df = df.drop("birth_year", axis=1) #Drop birth_year for clustering; consider it for interpretation
	df = df[df["first_policy"]<50000] #Drop one case where first_policy year <50000 
	df.reset_index(inplace=True,drop=True)
	df_num = pd.DataFrame(df[['first_policy', 'salary_year','mon_value','claims_rate','premium_motor','premium_household','premium_health','premium_life','premium_work_comp']])
	
	# Define individual multipliers for features
	thresholds = v
	outliers = []
	for col, th in thresholds.items():
		direct = "pos"
		if col == "mon_value":
			direct = "neg"
		outliers.append(get_outliers_i(df_num, col, th, direct))
	
	df_outlier = df.iloc[list(set([o for l in outliers for o in l]))]
	df = df[~df.index.isin(df_outlier.index.values)]
	
	# Filling nan values in premium columns
	#Assumption: nan values in premium mean no contract 
	df[["premium_motor","premium_household","premium_health","premium_life","premium_work_comp"]] = df[["premium_motor","premium_household","premium_health","premium_life","premium_work_comp"]].fillna(0)
	
	# Drop customers with nan values in "salary_year","educ" or "has_children" because these are only a few customers and we do not have any reasonable correlation to fill it 
	df_dropped = df[df[["salary_year","educ","has_children"]].isna().any(axis=1)]
	df = df.dropna(subset=["salary_year","educ","has_children"])
	
	######### Feature-engineering and -selection #########
	df.reset_index(drop=True, inplace=True)
	
	# Calculate total amount paid for premiums per year per customer
	df["premium_total"] = [sum(p for p in premiums if p > 0) for i, premiums in df[['premium_motor','premium_household','premium_health', 'premium_life','premium_work_comp']].iterrows()]
	# Split the features in customer- and product-related. 
	customer_related_num = ['salary_year', 'mon_value',  'claims_rate', 'premium_total'] # dont use first_policy because the clusters are clearer without
	
	######### Customer-related ##########
	################ K-Means with only numerical Features #################
	# Normalization
	scaler = StandardScaler()
	cust_norm = scaler.fit_transform(df[customer_related_num])
	df_cust_num_norm = pd.DataFrame(cust_norm, columns = customer_related_num)
		
	# Model fit
	kmeans_cust = KMeans(n_clusters=4, random_state=1).fit(df_cust_num_norm)
	# Model predict
	df["c_cluster"] = kmeans_cust.labels_
	silhouette_avg = silhouette_score(df_cust_num_norm, kmeans_cust.labels_)
	print("the average silhouette_score for " + str(v) + " is " + str(silhouette_avg)) 
	
	# Inverse Normalization for Interpretation
	cc = pd.DataFrame(scaler.inverse_transform(X=kmeans_cust.cluster_centers_), columns = customer_related_num)
	cluster_centroids.append(dict(key=v, value=cc))


