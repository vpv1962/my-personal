#!/usr/bin/env python
# coding: utf-8

# In[26]:


print('Coursera Week 3 Assignment')
print('Submitted by V.P.Verma')


# In[14]:


get_ipython().system('conda install -c conda-forge geopy --yes ')
# convert an address into latitude and longitude values
from geopy.geocoders import Nominatim


# In[22]:


import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import numpy as np
import json
import requests
from pandas.io.json import json_normalize
# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library

print('Libraries imported.')


# In[5]:


#Reads page and keeps tables with first row as header
dfs = pd.read_html("https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M", header=0)
#Keeps first table
pc_df = dfs[0]
pc_df.head()


# In[6]:


#Removes the rows with Borough Not Asiggned
NA = pc_df[ pc_df["Borough"] == "Not assigned" ].index
pc_df.drop(NA , inplace=True)
pc_df.head()


# In[7]:


#Checks if there is a Neighborhood Not assigned
na=pc_df["Neighborhood"]=="Not assigned"
na.value_counts()


# In[ ]:


## STEP 2


# In[8]:


#Calculate number of rows and colums
pc_df.shape


# In[9]:


pc_df.head()


# In[30]:


#Read the csv file with the  Postal codes and coordinates
coor=pd.read_csv("http://cocl.us/Geospatial_data")
coor.head()


# In[31]:


#Merge the dfs of Postal codes that we made in last question, with the one with the coordinates
#Function merge: we indicate each df and the column to compare
pc_coor=pd.merge(pc_df, coor, on="Postal Code")
pc_coor.head()


# In[32]:


pc_coor.shape


# In[33]:


##STEP 3


# In[34]:


#Check how many Neighborhoods are for heach Borough
pc_coor["Borough"].value_counts()


# In[35]:


#Keep only the Boroughs that contain "Toronto" in their names
tor_df=pc_coor[pc_coor["Borough"].str.contains('Toronto')]
tor_df.head()


# In[36]:


#Check that only keep the correct ones
tor_df["Borough"].value_counts()


# In[37]:


import requests # library to handle requests
CLIENT_ID = '4IT2ECHAKR0ZWI4UEVD1CKDMVGRSO2UF5YJMTNBPLOPIN2UJ' # your Foursquare ID
CLIENT_SECRET = 'KN40RYTBWZXKYNCEDEAFTZFXQW5ZLLKYR5T3OTYKYC2NGIHA' # your Foursquare Secret
LIMIT=100
VERSION = '20200618'

#As in the New York example, we create a function to explore all the neighborhoods in Toronto

def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(CLIENT_ID,CLIENT_SECRET,VERSION,lat,lng,radius,LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]["groups"][0]["items"]
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[38]:


toronto_venues = getNearbyVenues(names=tor_df['Neighborhood'],latitudes=tor_df['Latitude'],longitudes=tor_df['Longitude'])


# In[39]:


toronto_venues.head()


# In[40]:


toronto_venues.groupby('Neighborhood').count()


# In[41]:


# one hot encoding
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
toronto_onehot['Neighborhood'] = toronto_venues['Neighborhood'] 

# move neighborhood column to the first column
first = toronto_onehot['Neighborhood']
toronto_onehot.drop(labels=['Neighborhood'], axis=1,inplace = True)
toronto_onehot.insert(0, 'Neighborhood', first)
toronto_onehot.head()


# In[42]:


toronto_grouped = toronto_onehot.groupby('Neighborhood').mean().reset_index()
toronto_grouped


# In[43]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[44]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = toronto_grouped['Neighborhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# In[46]:


# add clustering labels
#neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)
toronto_merged = tor_df
# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')
toronto_merged.head()


# In[45]:


from sklearn.cluster import KMeans

# set number of clusters
kclusters = 5

toronto_grouped_clustering = toronto_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]


# In[ ]:


import pandas as pd
import folium
import matplotlib.cm as cm
import matplotlib.colors as colors

# create map
map_clusters = folium.Map(location=[43.651070, -79.347015], zoom_start=10)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighborhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)       
map_clusters

