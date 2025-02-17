#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Machine learning can uncover complex patterns in climate data to improve predictions of global warming's impacts. This analysis explores these techniques to better understand and mitigate future climate change.
# 
# ## What is Climate Change?
# 1) Change in climate patterns
# 2) Very apparent with the start of 20th Century
# 3) Attributed largely to the increased levels of atmospheric carbon dioxide produces by the use of fossil fuels
# ## Effects:
# 1) Temperature increase
# 2) Sea level rise (because of massive ice mass loss)
# 3) Shifts in flower's blooming
# 4) Extreme weather events
# ## Questions:
# 1) Did the world increased in temperature?
# 2) Where are the biggest increases?
# 3) By how much?

# In[1]:


#Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import datetime

# Plotly
import chart_studio.plotly as py
import plotly.express as px
import plotly.graph_objects as go
import colorlover as cl
from plotly.subplots import make_subplots

# Colors
#maroon = rgb(128,0,0)
#chocolate = rgb(210,105,30)
#sky blue = rgb(135,206,235)
#olive drab = rgb(107,142,35)
#steel blue = rgb(70,130,180)


# # 1.0 Data on years - overall land and ocean temperature change
# 1) Reading the data
# 2) Handling Missing data
# 3) Create new features (day, month, year etc.)
# 4) Analysing distribution of data in time to assess how to better analyse the numbers

# In[2]:


# Read the data
data = pd.read_csv("GlobalTemperatures.csv")
data.head(2)

# Make a copy of the data for future graphs
copy = data.copy()

# Missing values
data.isna().sum() # there are 1200 missing values for Max, Min and Land&Ocean Average Temp
plt.figure(figsize = (16, 3))
sns.heatmap(data.isna())


# In[3]:


# Missing data with listwise deletion
# Because data is missing in chucks and we are dealing with time series data, we will delete all rows that have at least 
# one missing value.

data.dropna(axis = 0, inplace = True)


# Dealing with the DATE
data['Date'] = pd.to_datetime(data.dt) # converted all dates to the same format

data2 = data.copy() # create a new dataset
data2.drop(columns = ['dt'], axis = 1, inplace = True) # drop the dt column

# Creating new features
data2['day'] = data2['Date'].dt.day
data2['week'] = data2['Date'].dt.isocalendar().week
data2['month'] = data2['Date'].dt.month
data2['year'] = data2['Date'].dt.year

# Week data is not evenly distributed
data2['week'].value_counts() # very uneven information on weeks

# For future analysis, we will work only on yearly data, as average (because there are dates missing and data is not consistent)
earth_data = data2.groupby(by = 'year')[['LandAverageTemperature', 'LandAverageTemperatureUncertainty',
       'LandMaxTemperature', 'LandMaxTemperatureUncertainty',
       'LandMinTemperature', 'LandMinTemperatureUncertainty',
       'LandAndOceanAverageTemperature',
       'LandAndOceanAverageTemperatureUncertainty']].mean().reset_index()

earth_data['turnpoint'] = np.where(earth_data['year'] <= 1975, 'before', 'after') # creating a new columns
earth_data.head(2)


# # 1.1 Summary Statistics 1850-2010
# Some very simple summary; we have a total of 166 observations. Now we'll go into visualisations, to finally get some insights.

# In[4]:


# Simple Summary Statistics
earth_data[['LandAverageTemperature', 'LandMaxTemperature', 
       'LandMinTemperature', 'LandAndOceanAverageTemperature']].describe()


# # 1.2 Plotly
# ## 1) Land Average Temperature from 1750 to 2010
# 
# Land Average Temperature had a steady increase: quite flat variation until mid 1800, and then started to increase slowly (when the first Industrial Revolution ended, and the effects were starting to show up on the environment). Somewhere in the 1970s the growth started to accelerate. The 1970s mark an increasing trend in machine automation, meaning a bust in foreign trading, so the production started to surge.

# In[5]:


# Creating the dataset - using copy
copy['Date'] = pd.to_datetime(copy.dt)
copy['year'] = copy['Date'].dt.year
land_avg = copy.groupby('year')[['LandAverageTemperature', 'LandAverageTemperatureUncertainty']].mean().reset_index()

# Creating the graph
fig = go.Figure()
fig.update_layout(title="Land Average Temperature: 1750-2010", title_font_size = 20,
                  font=dict( family="Courier New, monospace", size=12,color="#7f7f7f"),
                  template = "ggplot2", hovermode= 'closest')
fig.update_xaxes(showline=True, linewidth=1, linecolor='gray')
fig.update_yaxes(showline=True, linewidth=1, linecolor='gray')

fig.add_trace(go.Scatter(x = land_avg['year'], y = land_avg['LandAverageTemperature'], mode = 'lines',
                        name = 'Land Avg Temp', marker_color='rgb(128, 0, 0)'))

fig.add_trace(go.Scatter(x = land_avg['year'], y = land_avg['LandAverageTemperatureUncertainty'], mode = 'lines',
                        name = 'Land Avg Temp Error', marker_color = 'rgb(107,142,35)'))


# ## 2) When did Global Warming Started?
# The increase on all levels, in both land and ocean almost mirror one another. No doubt the Industrial Revolution had an effect between 1900 and 1975, but combining with the population increase that started to surge somewhere in 1975 (from ~2.5 bil in 1950 to 5 bil in 2000) created a much bigger negative contribution to the overall global warming state.

# In[6]:


# Figure layout
fig = make_subplots(rows=2, cols=2, insets=[{'cell': (1,1), 'l': 0.7, 'b': 0.3}])
fig.update_layout(title="When Global Warming Started?",font=dict( family="Courier New, monospace", size=12,color="#7f7f7f"),
                 template = "ggplot2", title_font_size = 20, hovermode= 'closest')
fig.update_xaxes(showline=True, linewidth=1, linecolor='gray')
fig.update_yaxes(showline=True, linewidth=1, linecolor='gray')

# Figure data
fig.add_trace(go.Scatter(x = earth_data['year'], y = earth_data['LandAverageTemperature'], mode = 'lines',
                        name = 'Land Avg Temp', marker_color='rgb(128, 0, 0)'), row = 1, col = 1)
fig.add_trace(go.Scatter( x=[1975, 1975], y=[7.5, 10], mode="lines",line=go.scatter.Line(color="gray"), showlegend=False),
             row = 1, col = 1)
#=============================================================================
fig.add_trace(go.Scatter(x = earth_data['year'], y = earth_data['LandMinTemperature'], mode = 'lines',
                        name = 'Land Min Temp', marker_color='rgb(210,105,30)'), row = 1, col = 2)
fig.add_trace(go.Scatter( x=[1975, 1975], y=[1.5, 4.5], mode="lines",line=go.scatter.Line(color="gray"), showlegend=False),
             row = 1, col = 2)
#=============================================================================
fig.add_trace(go.Scatter(x = earth_data['year'], y = earth_data['LandMaxTemperature'], mode = 'lines',
                        name = 'Land Max Temp', marker_color='rgb(135,206,235)'), row = 2, col = 1)
fig.add_trace(go.Scatter( x=[1975, 1975], y=[13, 15.5], mode="lines",line=go.scatter.Line(color="gray"), showlegend=False),
             row = 2, col = 1)
#=============================================================================
fig.add_trace(go.Scatter(x = earth_data['year'], y = earth_data['LandAndOceanAverageTemperature'], mode = 'lines',
                        name = 'Land&Ocean Avg Temp', marker_color='rgb(107,142,35)'), row = 2, col = 2)
fig.add_trace(go.Scatter( x=[1975, 1975], y=[14.5, 16], mode="lines",line=go.scatter.Line(color="gray"), showlegend=False),
             row = 2, col = 2)


# ## 3) Change in Average Temperature before/after 1975 - cold is hotter?
#  compare temperatures before and after.
# 
# 1) Land Average Temperature - an increase from 8.37 degrees to 9.20; almost 1 full grade
# 2) Land Minimum Temperatures - an increase from 2.45 degrees to 3.64; 1.19 grades increase
# 3) Land Maximum Temperature - an increase from 14.18 degrees to 14.89; 0.71 grades increase
# 4) Land and Ocean Average Temperature - an increase from 15.06 degrees to 15.62; 0.56 grades increase
# 
# 
# 

# In[7]:


# Change in average temperature before/after 1975

# Figure layout
fig = make_subplots(rows=2, cols=2, insets=[{'cell': (1,1), 'l': 0.7, 'b': 0.3}])
fig.update_layout(title="Average Temperatures before and after 1975",font=dict( family="Courier New, monospace", size=12,color="#7f7f7f"),
                 template = "ggplot2", title_font_size = 20, hovermode= 'closest')

# Figure data
fig.add_trace(go.Box(x = earth_data['LandAverageTemperature'], y = earth_data['turnpoint'],boxpoints = 'all',jitter = 0.3, 
                     pointpos = -1.6, marker_color = 'rgb(128, 0, 0)', boxmean = True, name = 'Land Avg Temp'),
             row = 1, col = 1)
fig.add_trace(go.Box(x = earth_data['LandMinTemperature'], y = earth_data['turnpoint'],boxpoints = 'all',jitter = 0.3, 
                     pointpos = -1.6, marker_color = 'rgb(210,105,30)', boxmean = True, name = 'Land Min Temp'),
              row = 1, col = 2)
fig.add_trace(go.Box(x = earth_data['LandMaxTemperature'], y = earth_data['turnpoint'],boxpoints = 'all',jitter = 0.3, 
                     pointpos = -1.6, marker_color = 'rgb(135,206,235)', boxmean = True, name = 'Land Max Temp'),
             row = 2, col = 1)
fig.add_trace(go.Box(x = earth_data['LandAndOceanAverageTemperature'], y = earth_data['turnpoint'], boxpoints = 'all',jitter = 0.3, 
                     pointpos = -1.6, marker_color = 'rgb(107,142,35)', boxmean = True, name = 'Land&Ocean Avg Temp'),
             row = 2, col = 2)


fig.update_traces(orientation='h')


# ## 2.0 Data Geographical View
# Let's look at the countries now

# In[8]:


import pandas as pd

# Read the file
countries = pd.read_csv("GlobalLandTemperaturesByCity.csv")

# Convert date column to datetime
countries['Date'] = pd.to_datetime(countries['dt'], errors='coerce')
countries['year'] = countries['Date'].dt.year

# Ensure Latitude & Longitude are numeric
countries['Latitude'] = countries['Latitude'].str.extract(r'([-+]?[0-9]*\.?[0-9]+)').astype(float)
countries['Longitude'] = countries['Longitude'].str.extract(r'([-+]?[0-9]*\.?[0-9]+)').astype(float)

# Drop NaN values in key columns
countries = countries.dropna(subset=['year', 'City', 'Country', 'Latitude', 'Longitude'])

# Remove duplicate columns if any
countries = countries.loc[:, ~countries.columns.duplicated()]

# Keep only numeric columns for mean calculation, excluding Latitude & Longitude
numeric_cols = countries.select_dtypes(include=['number']).columns
agg_dict = {col: 'mean' for col in numeric_cols if col not in ['Latitude', 'Longitude']}
agg_dict['Latitude'] = 'first'  # Keep first value of Latitude
agg_dict['Longitude'] = 'first'  # Keep first value of Longitude

# Group by year, City, and Country while applying correct aggregations
by_year = countries.groupby(['year', 'City', 'Country'], as_index=False).agg(agg_dict)

# Append the continent & ISO codes
continent_map = pd.read_csv("continents2.csv")
continent_map['Country'] = continent_map['name']
continent_map = continent_map[['Country', 'region', 'alpha-2', 'alpha-3']]

# Merge the continent data
data = pd.merge(left=by_year, right=continent_map, on='Country', how='left')

# Filter data from 1825 onwards
data = data[data['year'] >= 1825]

# Create datasets without aggregation errors (grouping by region and year for means)
region = data.dropna(subset=['region']).groupby(['region', 'year'], as_index=False).mean(numeric_only=True)
countries = data.dropna(subset=['region', 'Country']).groupby(['region', 'Country', 'year'], as_index=False).mean(numeric_only=True)
cities = data.dropna(subset=['region', 'City']).groupby(['region', 'Country', 'City', 'year'], as_index=False).mean(numeric_only=True)

# Display results
print(data.head())


# ## 2.1 Plotly
# ### 1) Continents Increase in Average Temperature
# There are some variations at the beginning of the period, mainly due to a fluctuation in the gathering of data (there are some countries where there is info in some years and some years there is not.
# 
# Anyhow, all continents look to have increased in Temperature in about the same pace.

# In[9]:


# Figure layout
fig = make_subplots(rows=1, cols=2, insets=[{'cell': (1,1), 'l': 0.7, 'b': 0.3}])
fig.update_layout(title="Continents increase in Average Temperature", title_font_size = 20,
                  font=dict( family="Courier New, monospace", size=12,color="#7f7f7f"),
                  template = "ggplot2", hovermode= 'closest')
fig.update_xaxes(showline=True, linewidth=1, linecolor='gray')
fig.update_yaxes(showline=True, linewidth=1, linecolor='gray')

#============================= Scatter =============================
fig.add_trace(go.Scatter(x = region[region['region'] == 'Europe']['year'], y = region[region['region'] == 'Europe']['AverageTemperature'], mode = 'lines',
                        name = 'Europe', marker_color='rgb(128, 0, 0)'), row = 1, col = 1)
fig.add_trace(go.Scatter(x = region[region['region'] == 'Americas']['year'], y = region[region['region'] == 'Americas']['AverageTemperature'], mode = 'lines',
                        name = 'Americas', marker_color='rgb(210,105,30)'), row = 1, col = 1)
fig.add_trace(go.Scatter(x = region[region['region'] == 'Asia']['year'], y = region[region['region'] == 'Asia']['AverageTemperature'], mode = 'lines',
                        name = 'Asia', marker_color='rgb(135,206,235)'), row = 1, col = 1)
fig.add_trace(go.Scatter(x = region[region['region'] == 'Africa']['year'], y = region[region['region'] == 'Africa']['AverageTemperature'], mode = 'lines',
                        name = 'Africa', marker_color='rgb(107,142,35)'), row = 1, col = 1)
fig.add_trace(go.Scatter(x = region[region['region'] == 'Oceania']['year'], y = region[region['region'] == 'Oceania']['AverageTemperature'], mode = 'lines',
                        name = 'Oceania', marker_color='rgb(70,130,180)'), row = 1, col = 1)

#============================= Bar =============================
y1 = np.round(region.groupby(by = 'region')['AverageTemperature'].mean().tolist(), 1)
y2 = np.round(region.groupby(by = 'region')['AverageTemperature'].max().tolist(), 1)

fig.add_trace(go.Bar(x = region['region'].unique(), y = region.groupby(by = 'region')['AverageTemperature'].mean().tolist(), 
                     name = 'Mean Temp', marker_color = 'rgb(188,143,143)', text = y1, textposition = 'auto'),
              row = 1, col = 2)
fig.add_trace(go.Bar(x = region['region'].unique(), y = region.groupby(by = 'region')['AverageTemperature'].max().tolist(), 
                     name = 'Max Temp', marker_color = 'rgb(222,184,135)', text = y2, textposition = 'auto'),
              row = 1, col = 2)


# ### 2) Countries rank - from the biggest to smallest increase in temperature since 1825
# 
# 1) Brazil and Argentina - BIG deforestation issues (their wildfires have increased big time and the agriculture is the main factor)
# 2) Kazakhstan - place for testing biological and nuclear weapons by the soviets. Also, here are located the most polluting industries. Most of their water is infected by industrial and agricultural runoff and it is in some places radioactivity. (http://factsanddetails.com/central-asia/Kazakhstan/sub8_4f/entry-4681.html)
# 3) Turkmenistan - Desertification and drying of the Aral Sea, due to HEAVY agricultural practices. (https://en.wikipedia.org/wiki/Environmental_issues_in_Turkmenistan)
# 4) Nepal - Air Pollution. 1 in 10 Nepalese suffer from chronic lung problem, and the life expectancy of a new born is shorter by 2 years due solely to problems air quality related. (https://www.nepalitimes.com/here-now/air-pollution-is-more-dangerous-than-smoking/)

# In[10]:


# Data
mean = countries.groupby(['Country', 'region'])['AverageTemperature'].mean().reset_index()
maximum = countries.groupby(['Country', 'region'])['AverageTemperature'].max().reset_index()

difference = pd.merge(left = mean, right = maximum, on = ['Country', 'region'])
difference['diff'] = difference['AverageTemperature_y'] - difference['AverageTemperature_x']

# Graph
fig = go.Figure()
fig.update_layout(title="Difference in Temperature (Countries)", title_font_size = 20,
                  font=dict( family="Courier New, monospace", size=13,color="#7f7f7f"),
                  template = "ggplot2", autosize = False, height = 3500, width = 750)
fig.update_xaxes(showline=True, linewidth=1, linecolor='gray')
fig.update_yaxes(showline=True, linewidth=1, linecolor='gray')

sort_diff = difference[['Country', 'region', 'diff']].sort_values(by = 'diff', ascending = True)
fig.add_trace(go.Bar(x = sort_diff['diff'], y = sort_diff['Country'], orientation = 'h',
                    marker=dict(color='rgb(222,184,135)', line=dict( color='rgb(188,143,143)', width=0.6))))
fig.show()


# ### 2.2 Plotly Maps
# ### 1) Interactive Globe Map - Temperature increase over the years
# First years have some fluctuations due to absence of data for some nations, but the biggest increases are visible in Europe, South America and West Asia.

# In[11]:


import pandas as pd
import plotly.express as px

# Select numeric columns
numeric_cols = data.select_dtypes(include=['number']).columns

# Drop rows where required columns are missing
data_clean = data.dropna(subset=['region', 'Country', 'year', 'alpha-3'])

# Group by region, country, year, and alpha-3 and calculate mean for numeric columns
map_countries = (
    data_clean
    .groupby(['region', 'Country', 'year', 'alpha-3'], as_index=False)[numeric_cols]
    .mean()
)

# Min temperature adjustment
map_countries['AverageTemperature'] = map_countries['AverageTemperature'] + 6

# Drop NaN values in AverageTemperature before plotting
map_countries = map_countries.dropna(subset=['AverageTemperature'])

# Plot the interactive globe map
fig = px.scatter_geo(
    map_countries, locations='alpha-3', color='region',
    color_discrete_sequence=['rgb(128,0,0)', 'rgb(210,105,30)', 'rgb(135,206,235)', 'rgb(107,142,35)'],
    hover_name="Country", size="AverageTemperature", size_max=15, opacity=0.8,
    animation_frame="year",
    projection="natural earth", title='Interactive Globe Map - Temperature Increase'
)

fig.show()


# ## 2. Globe Map - difference between the mean and maximum temperature
# the bubbles - countries
# the COLOR of the bubbles - the mean temperature
# the size of the bubbles - how much the temperature increased?
# What do we see? The biggest temperature changes were in Europe and South America.
# 
# There is again the question, as we see that more colder countries (situated to the North) had a biggest increase in Temperature than the overall South, why is this happening?

# In[12]:


# Calculating the difference column
mean = map_countries.groupby(['region','Country','alpha-3'])['AverageTemperature'].mean().reset_index()
maximum = map_countries.groupby(['region','Country','alpha-3'])['AverageTemperature'].max().reset_index()
difference = pd.merge(left = mean, right = maximum, on = ['region','Country','alpha-3'])
difference['diff'] = difference['AverageTemperature_y'] - difference['AverageTemperature_x']
difference.rename(columns = {'AverageTemperature_y':'Maximum Average Temperature',
                             'AverageTemperature_y':'Overall Avg Temp'}, inplace = True)

# Figure
fig = px.scatter_geo(difference, locations="alpha-3", color="Overall Avg Temp", #we color by average temp
                     hover_name="Country", size="diff", size_max=15, #we size by how big is the difference
                     projection="natural earth", opacity = 0.8,
                     color_continuous_scale=('#283747', '#2874A6', '#3498DB', '#F5B041', '#E67E22', '#A93226'),
                     title = 'Global Map - difference between the mean and max temperatures')
fig.show()


# In[13]:


import os
print(os.getcwd())


# In[14]:


get_ipython().run_line_magic('pwd', '')

