import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.io import write_image

df = pd.read_csv("data/insurance_clusters.csv", index_col=0)

cust_num = ['salary_year', 'mon_value', 'claims_rate', 'premium_total']
prod_num = ['premium_motor', 'premium_household', 'premium_health', 'premium_life', 'premium_work_comp']
cat = [ 'educ', 'location', 'has_children', 'cancelled_contracts', 'nr_contracts']

c_cust_num_mean = df[cust_num + ["c_cluster"]].groupby("c_cluster").mean()
c_cat_mode = df[cat + ["c_cluster"]].groupby("c_cluster").agg(lambda x:x.value_counts().index[0])

# Comparing normalized centroids of clusters
scaler = StandardScaler()
df_norm = pd.DataFrame(scaler.fit_transform(df[cust_num]), columns=cust_num)
df_norm["cluster"] = df.loc[:,"c_cluster"]
c_cust_num_mean_norm = df_norm.groupby("cluster").mean()

# Create traces for centroids 
data = []
symbols = ["star", "star-square", "circle", "cross"]
for i in range(0,len(c_cust_num_mean_norm.index.values)):
	# centroids
	data.append(go.Scatter(y=[i for i in range(0,len(c_cust_num_mean_norm.columns.values))], x=c_cust_num_mean_norm.iloc[i,:], 
						mode='markers', name="Cluster " + str(i+1),
						marker=dict(size=12,color = i+1, colorscale="Viridis", symbol=symbols[i])
						))

layout = go.Layout(title=dict(text="Comparing centroids", y=0.9, x=0.5, xanchor="center", yanchor="top", font=dict(size=30)),
				   yaxis=dict(tickmode = 'array',tickvals = [k for k in range(0, len(c_cust_num_mean_norm.columns.values))],
						ticktext = [col for col in c_cust_num_mean_norm.columns.values],
						tickfont = dict(size=20)), xaxis=dict(title="Normalized values"), template="plotly_white")
fig = go.Figure(data=data, layout=layout)
fig.write_image("centroids_compared.png", width=1200, height=500)


# Leverage
sum_mon_val = df[["mon_value", "c_cluster"]].groupby("c_cluster").sum()
prop_mon_val = sum_mon_val["mon_value"].apply(lambda x: x/sum(sum_mon_val["mon_value"]))
count_cust = df["c_cluster"].value_counts().sort_index()
prop_cust= count_cust.apply(lambda x: x/sum(count_cust.values))

mon_labels = [str(round(p*100,2)) + "%" for p in prop_mon_val.values]
cus_labels = [str(round(p*100,2)) + "%" for p in prop_cust.values]
leverage = prop_mon_val / prop_cust

data = [go.Bar(name="Monetary value", x=prop_mon_val.index, y=prop_mon_val, text=mon_labels,textposition="auto"),
		go.Bar(name="Proporation of customers", x=prop_mon_val.index, y=prop_cust, text=cus_labels,textposition="auto")]


layout = go.Layout(title=dict(text="Leverage", y=0.9, x=0.5, xanchor="center", yanchor="top", font=dict(size=30)),
				   xaxis=dict(tickmode = 'array', tickvals=[k for k in range(0, len(prop_cust.index.values))],
						ticktext = ["Cluster " + str(c + 1) for c in prop_cust.index.values],
						tickfont = dict(size=20)), template="plotly_white")

fig = go.Figure(data=data, layout=layout)
for i in range(0,4):
	fig.add_annotation(
	    go.layout.Annotation(
	            x=i,
	            y=0.7,
	            text=round(leverage[i],2), 
				font=dict(size=16),
		        align="center",bordercolor="#c7c7c7",
		        borderwidth=2,
		        borderpad=4, 
				showarrow=False))

fig.write_image("leverage.png", width=1000, height=500)








