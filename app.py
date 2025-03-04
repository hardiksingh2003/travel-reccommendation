import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Load the datasets
hotels_df = pd.read_csv('hotels.csv')
df_combined = pd.read_csv('df_combined.csv')
df_flights = pd.read_csv('flights.csv')
df_restaurants = pd.read_csv('restaurant.csv')
df_trains = pd.read_csv('trains.csv')

df_restaurants['latitude'] = pd.to_numeric(df_restaurants['latitude'], errors='coerce')
df_restaurants['longitude'] = pd.to_numeric(df_restaurants['longitude'], errors='coerce')
df_restaurants['Average Cost for two'] = pd.to_numeric(df_restaurants['Average Cost for two'], errors='coerce')

# Geoapify API Key
API_KEY = "eec49d18814f4e1687370a8b751c0937"  # Replace with your actual API key

def get_lat_lng(address):
    url = f"https://api.geoapify.com/v1/geocode/search?text={address}&limit=1&apiKey={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data["features"]:
            result = data["features"][0]
            return result["geometry"]["coordinates"][1], result["geometry"]["coordinates"][0]
    return None, None

def find_cluster(latitude, longitude, df):
    cluster_centroids = df.groupby('cluster')[['latitude', 'longitude']].mean()
    neighbors = NearestNeighbors(n_neighbors=1)
    neighbors.fit(cluster_centroids)
    _, nearest_cluster = neighbors.kneighbors([[latitude, longitude]])
    return cluster_centroids.index[nearest_cluster[0][0]]

def recommend_hotels_based_on_query(query, cluster_df, top_n=3):
    data_with_query = cluster_df['hotel_info'].tolist() + [query]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data_with_query)
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    cluster_df['similarity_score'] = cosine_similarities
    return cluster_df.sort_values(by='similarity_score', ascending=False).head(top_n)[['property_name', 'guest_recommendation', 'address']]

def recommend_restaurants(cuisine_query, cluster_df, top_n=3):
    vectorizer = TfidfVectorizer()
    cuisine_data = cluster_df['Cuisines'].tolist() + [cuisine_query]
    tfidf_matrix = vectorizer.fit_transform(cuisine_data)
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    cluster_df['similarity_score'] = cosine_similarities
    cluster_df['weighted_score'] = (
        cluster_df['similarity_score'] * 0.5 + (cluster_df['rating_integer'] / 5) * 0.3 - 
        (cluster_df['Average Cost for two'] / cluster_df['Average Cost for two'].max()) * 0.2
    )
    return cluster_df.sort_values(by='weighted_score', ascending=False).head(top_n)[['Restaurant Name', 'rating_integer', 'Average Cost for two']]

st.title("Travel Recommendation System")

address = st.text_input("Enter your address:")
query = st.text_input("Enter hotel preferences:")
cuisine_query = st.text_input("Enter preferred cuisine:")
origin = st.selectbox("Select flight origin:", df_flights['origin'].unique())
destination = st.selectbox("Select flight destination:", df_flights['destination'].unique())
day = st.selectbox("Select flight day:", ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
source = st.selectbox("Select train source:", df_trains['Source_Station_Name'].unique())
destination_station = st.selectbox("Select train destination:", df_trains['Destination_Station_Name'].unique())
train_day = st.selectbox("Select train day:", ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])

if st.button("Search"):
    latitude, longitude = get_lat_lng(address)
    if latitude and longitude:
        target_cluster_hotels = find_cluster(latitude, longitude, hotels_df)
        cluster_df_hotels = df_combined[df_combined['cluster'] == target_cluster_hotels]
        hotel_recommendations = recommend_hotels_based_on_query(query, cluster_df_hotels)
        
        target_cluster_restaurants = find_cluster(latitude, longitude, df_restaurants)
        cluster_df_restaurants = df_restaurants[df_restaurants['cluster'] == target_cluster_restaurants]
        restaurant_recommendations = recommend_restaurants(cuisine_query, cluster_df_restaurants)
        
        flight_results = df_flights[(df_flights['origin'] == origin) & (df_flights['destination'] == destination) & (df_flights['dayOfWeek'].str.contains(day))].head(5)
        
        train_results = df_trains[(df_trains['Source_Station_Name'] == source) & (df_trains['Destination_Station_Name'] == destination_station) & (df_trains['days'] == train_day)]

        st.subheader("Hotel Recommendations")
        st.dataframe(hotel_recommendations)
        
        st.subheader("Restaurant Recommendations")
        st.dataframe(restaurant_recommendations)
        
        st.subheader("Flight Options")
        st.dataframe(flight_results[['airline', 'scheduledDepartureTime', 'scheduledArrivalTime', 'flightNumber']])
        
        st.subheader("Train Options")
        st.dataframe(train_results[['Train_No', 'Train_Name', 'days']])
    else:
        st.error("Could not find location. Please try again.")
