import pandas as pd
from bokeh.core.properties import value
from bokeh.io import show, output_notebook
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Legend, Label, HoverTool, FactorRange, Div, NumeralTickFormatter, CategoricalTicker, LinearColorMapper, ColorBar, PrintfTickFormatter, BasicTicker, SingleIntervalTicker, LinearAxis
from bokeh.palettes import Spectral, RdYlGn
from bokeh.transform import factor_cmap, dodge, transform
from bokeh.layouts import gridplot
from bokeh.layouts import row as Row
from bokeh.layouts import column as Column

df = pd.read_csv("A2Z Insurance.csv")

# Preprocessing
df = df.set_index("Customer Identity")
newnames = ["first_policy","birth_year","educ","salary_monthly","location","has_children","mon_value","claims_rate","premium_motor","premium_household","premium_health","premium_life","premium_work_comp"]
df.rename(columns=dict(zip(df.columns.values, newnames)), inplace = True)
df["salary_monthly"] = df["salary_monthly"]*12 #multiple salary by 12 to have everything in the same unit (year)
df.rename(columns={"salary_monthly":"salary_year"}, inplace = True)
df = df.drop("birth_year", axis=1) #Drop birth_year for clustering; consider it for interpretation
df = df[df["first_policy"]<50000] #Drop one case where first_policy year <50000 






def create_boxplot(df, col, multi):
	feature = pd.DataFrame(df.loc[:,col])
	q1 = feature[col].quantile(q=0.25)
	q2 = feature[col].quantile(q=0.5)
	q3 = feature[col].quantile(q=0.75)
	median = feature[col].median()
	mean = feature[col].mean()
	feature["q1"] = q1
	feature["q2"] = q2
	feature["q3"] = q3
	feature["median"] = median
	feature["mean"] = mean
	feature["iqr"] = q3 - q1
	feature["upper"] = q3 + multi*feature["iqr"]
	feature["lower"] = q1 - multi*feature["iqr"]
	feature["upper5"] = q3 + 5*feature["iqr"]
	feature["lower5"] = q1 - 5*feature["iqr"]
	feature["upper10"] = q3 +10*feature["iqr"]
	feature["lower10"] = q1 - 10*feature["iqr"]
	feature["lower20"] = q1 - 20*feature["iqr"]
	feature["upper20"] = q3 + 20*feature["iqr"]
	
	feature["countQQ"] = len(feature[(feature["claims_rate"] >= feature["q1"]) & (feature["claims_rate"] <= feature["q3"])])

	
	cds=ColumnDataSource(feature)

	p_a = figure(plot_width=550, x_range=(0,2),y_range=(feature["lower20"].values[0] - feature["iqr"].values[0],feature["upper20"].values[0] + feature["iqr"].values[0]), toolbar_location=None, 
	           title= "Boxplot for " + col + " with " + str(multi) + "x multiplier")
	
	# stems
	p_a.segment(x0=1, y0="upper20", x1=1, y1="q3", line_color="black", source=cds)
	p_a.segment(x0=1, y0="lower20", x1=1, y1="q1", line_color="black", source=cds)
	
	# boxes
	p_a.vbar(1, 0.8, "q2", "q3", fill_color=Spectral[9][3], line_color="black", source=cds)
	p_a.vbar(1, 0.8, "q1", "q2", fill_color=Spectral[9][2], line_color="black", source=cds)
	p_a.text(x=1.5, y=feature["median"].values[0], text="countQQ", text_align="center", text_font_style="bold")
	
	# whiskers
	p_a.rect(1, "lower", 0.2, 0.01, line_color="black", source=cds)
	p_a.rect(1, "upper", 0.2, 0.01, line_color="black", source=cds)
	p_a.rect(1, "lower5", 0.2, 0.01, line_color="black", source=cds)
	p_a.rect(1, "upper5", 0.2, 0.01, line_color="black", source=cds)
	p_a.rect(1, "lower10", 0.2, 0.01, line_color="black", source=cds)
	p_a.rect(1, "upper10", 0.2, 0.01, line_color="black", source=cds)
	p_a.rect(1, "lower20", 0.2, 0.01, line_color="black", source=cds)
	p_a.rect(1, "upper20", 0.2, 0.01, line_color="black", source=cds)
	
	# outliers 
	#p_a.circle(1, feature[((feature[col] > feature["upper"]) | (feature[col] < feature["lower"])) & (feature[col] < feature["highest"]) & (feature[col] > feature["lowest"])].index.values, size=10, color="navy", alpha=0.5 )

	
	p_a.xgrid.grid_line_color = None
	p_a.ygrid.grid_line_color = "white"
	p_a.grid.grid_line_width = 2
	p_a.xaxis.major_label_text_font_size="12pt"
	p_a.title.text_font_size = '18pt'
	
	layout = p_a
	show(layout)

feature = create_boxplot(df, "claims_rate", 5)