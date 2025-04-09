import os
import sys
import numpy as np
from datetime import datetime

# Fix import path BEFORE importing from 'core'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from core.recommender import RecipeRecommender
from core.model_pipeline import RecipeModelPipeline
import requests
from bs4 import BeautifulSoup
import json
import urllib.parse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Theme & Title
st.markdown("## 🍽️ AI-Powered Recipe Recommender")
st.markdown("*Get delicious suggestions tailored to your taste!*")
st.markdown("---")

# Config
DATA_PATH = "data/recipes_cleaned.csv"
MODEL_PATH = "data/ncf_model.pth"

@st.cache_resource
def load_recommender():
    return RecipeRecommender(model_path=MODEL_PATH, data_path=DATA_PATH)

@st.cache_resource
def load_pipeline():
    return RecipeModelPipeline(data_path=DATA_PATH)

recommender = load_recommender()
pipeline = load_pipeline()
data = pd.read_csv(DATA_PATH)
authors = sorted(data["author"].dropna().unique())
titles = sorted(data["title"].dropna().unique())
log_entries = []  # For tracking clicks
triggered_recommendation = None  # Store recipe clicked

# Sidebar
st.sidebar.title("🔍 Recipe Settings")
selected_author = st.sidebar.selectbox("Choose an author (as user):", authors)
top_n = st.sidebar.slider("Number of recommendations:", min_value=1, max_value=10, value=6)
model_type = st.sidebar.radio("Select Recommendation Model:", ["Neural Collaborative Filtering", "Content-Based Filtering", "Collaborative Filtering"])

# YouTube Helper
def get_youtube_details(query):
    search_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"
    response = requests.get(search_url)
    html = response.text
    start = html.find('var ytInitialData = ') + len('var ytInitialData = ')
    end = html.find(';</script>', start)
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(html[start:end])
        videos = data['contents']['twoColumnSearchResultsRenderer']['primaryContents']['sectionListRenderer']['contents'][0]['itemSectionRenderer']['contents']
        for video in videos:
            if 'videoRenderer' in video:
                video_id = video['videoRenderer']['videoId']
                thumbnail = video['videoRenderer']['thumbnail']['thumbnails'][-1]['url']
                return {
                    'url': f"https://www.youtube.com/watch?v={video_id}",
                    'thumbnail': thumbnail
                }
    except Exception as e:
        print("YouTube parse error:", e)
        return None

# Search Bar with Autocomplete
search_selection = st.selectbox("🔎 Search for a recipe (autocomplete enabled):", [""] + titles, index=0)
if search_selection:
    st.markdown(f"### 🔍 Search Results for: '{search_selection}'")
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data["title"])
    query_vec = tfidf.transform([search_selection])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    search_results = data.iloc[top_indices]["title"].tolist()

    cols = st.columns(3)
    for idx, title in enumerate(search_results):
        col = cols[idx % 3]
        with col:
            st.markdown(f"### 🍲 {title}")
            video_info = get_youtube_details(f"{title} recipe")
            if video_info:
                st.image(video_info['thumbnail'], use_container_width=True)
                video_url = video_info['url']
                st.markdown(f"[▶️ Watch Video]({video_url})", unsafe_allow_html=True)
                share_text = f"Check out this recipe: {title} - {video_url}"
                whatsapp_url = f"https://wa.me/?text={urllib.parse.quote(share_text)}"
                st.markdown(f"[📤 Share on WhatsApp]({whatsapp_url})", unsafe_allow_html=True)
                if st.button(f"👀 View Recipe: {title}", key=f"search_click_{idx}"):
                    log_entries.append({"user": selected_author, "title": title, "timestamp": datetime.now()})
                    triggered_recommendation = title
            else:
                st.info("No video found")
            st.markdown("---")

# Trigger new recommendations if recipe clicked
if triggered_recommendation:
    st.markdown(f"### 📌 Because you clicked on: {triggered_recommendation}")
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data["title"])
    query_vec = tfidf.transform([triggered_recommendation])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    related_titles = data.iloc[top_indices]["title"].tolist()

    cols = st.columns(3)
    for idx, title in enumerate(related_titles):
        col = cols[idx % 3]
        with col:
            st.markdown(f"### 🍲 {title}")
            video_info = get_youtube_details(f"{title} recipe")
            if video_info:
                st.image(video_info['thumbnail'], use_container_width=True)
                video_url = video_info['url']
                st.markdown(f"[▶️ Watch Video]({video_url})", unsafe_allow_html=True)
                share_text = f"Check out this recipe: {title} - {video_url}"
                whatsapp_url = f"https://wa.me/?text={urllib.parse.quote(share_text)}"
                st.markdown(f"[📤 Share on WhatsApp]({whatsapp_url})", unsafe_allow_html=True)
            else:
                st.info("No video found")
            st.markdown("---")

# RMSE Comparison Button
st.markdown("---")
if st.button("Compare Model RMSEs"):
    ncf_rmse, cb_rmse, cf_rmse = pipeline.run_pipeline()
    st.markdown("### 📊 Model RMSE Comparison")
    st.table({
        "Model": ["NCF", "Content-Based", "Collaborative Filtering"],
        "RMSE": [ncf_rmse, cb_rmse, cf_rmse]
    })

# Show Interaction Log
if log_entries:
    st.markdown("### 🧾 Click Tracking Log")
    log_df = pd.DataFrame(log_entries)
    st.dataframe(log_df, use_container_width=True)
