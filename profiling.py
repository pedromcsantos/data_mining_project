import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.graph_objs as go
import plotly.offline as pyo

df = pd.read_csv("insurance_clusters.csv", index_col = 0)

categoricals = ["educ","location","has_children","cancelled_contracts","has_all"]
numericals = ["salary_year","mon_value","claims_rate","premium_motor","premium_household","premium_health","premium_life","premium_work_comp","premium_total"]
clusters =  ["p_cluster","c_cluster"]

customer_related_num = ['salary_year',  'mon_value', 'claims_rate', 'premium_total', "c_cluster"]
customer_related_cat = ['location','has_children', 'educ', 'cancelled_contracts', 'has_all']
product_related_num = ["premium_motor","premium_household","premium_health","premium_life","premium_work_comp", "p_cluster"]

customer_cluster = df[customer_related_num].groupby("c_cluster").mean()
product_cluster = df[product_related_num].groupby("p_cluster").mean()
#### Checking if clusters overlap
## create a table/Matrix
cluster_matrix = pd.crosstab(df["p_cluster"], df["c_cluster"])
#plot
trace1 = go.Bar(x= cluster_matrix.index, y = cluster_matrix.loc[:,0], name = "c_cluster_0") 
trace2=go.Bar(x= cluster_matrix.index, y = cluster_matrix.loc[:,1], name = "c_cluster_1") 
trace3=go.Bar(x= cluster_matrix.index, y = cluster_matrix.loc[:,2], name = "c_cluster_2")
trace4=go.Bar(x= cluster_matrix.index, y = cluster_matrix.loc[:,3], name = "c_cluster_3")  
data = [trace1, trace2,trace3, trace4]
layout = go.Layout(title = "Cluster Combination", template = "plotly_dark", xaxis=dict(title="Product Cluster"), yaxis=dict(title="Frequency"))
fig = go.Figure(data = data, layout = layout)

fig.write_image("Cluster_combination.png", width=1200, height=500)

#profiling categoricals + numericals
for i in df[categoricals]:
    data = []
    for b in range(len(df[i].unique())):
        trace1 = go.Bar(x= df["c_cluster"].unique().sort(), y = (pd.crosstab(df[i], df["c_cluster"])/pd.crosstab(df[i], df["c_cluster"]).sum()).iloc[b,:], name = b) 
        data.append(trace1)
    layout = go.Layout(title = i, template = "plotly_dark", xaxis=dict(title="clusters"), yaxis=dict(title="Relative Freq."))
    fig = go.Figure(data = data, layout = layout)
    fig.update_layout(barmode = "stack")
    fig.write_image(i+".png",width=1200, height=500)
