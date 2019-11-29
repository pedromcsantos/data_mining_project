import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.offline import plot




df = pd.read_csv("insurance_clusters.csv", index_col=0)
cust_num = ['birth_year', 'salary_year', 'mon_value', 'claims_rate','customer_since', 'premium_total']
prod_num = ['premium_motor', 'premium_household', 'premium_health', 'premium_life', 'premium_work_comp']
cat = [ 'educ', 'location', 'has_children', 'is_family_policy', 'cancelled_contracts', 'nr_contracts']

c_cust_num_mean = df[cust_num + ["c_cluster"]].groupby("c_cluster").mean()
c_cat_mode = df[cat + ["c_cluster"]].groupby("c_cluster").agg(lambda x:x.value_counts().index[0])

# Comparing normalized centroids of clusters
scaler = StandardScaler()
cust_norm = scaler.fit_transform(c_cust_num_mean)
c_cust_num_mean_norm = pd.DataFrame(cust_norm, columns = c_cust_num_mean.columns)

data = []
for i in range(1,len(c_cust_num_mean_norm.columns.values)):
	data.append(go.Scatter(y=[i,i,i,i], x=c_cust_num_mean_norm.iloc[:,i], mode='markers+text', text=["C1","C2","C3","C4"], 
						marker=dict(size=20, color =  ["rgba(0,127,0,1.0)", "rgba(0,0,255,1.0)", "rgba(255,0,0,1.0)", "rgba(0,191,191,1.0)"])))
#layout = go.Layout(yaxis=dict(range=(0,4)))
fig = go.Figure(data=data)
plot(fig, auto_open=True)