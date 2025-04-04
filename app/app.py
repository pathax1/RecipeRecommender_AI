import os
import sys

# Fix import path BEFORE importing from 'core'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from core.recommender import RecipeRecommender
from youtubesearchpython import VideosSearch
import urllib.parse


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

recommender = load_recommender()
data = pd.read_csv(DATA_PATH)
authors = sorted(data["author"].dropna().unique())

# Sidebar
st.sidebar.title("🔍 Recipe Settings")
selected_author = st.sidebar.selectbox("Choose an author (as user):", authors)
top_n = st.sidebar.slider("Number of recommendations:", min_value=1, max_value=10, value=6)

# YouTube Helper
def get_youtube_details(query):
    search = VideosSearch(query, limit=1)
    result = search.result()
    try:
        video = result['result'][0]
        return {
            'url': video['link'],
            'thumbnail': video['thumbnails'][0]['url']
        }
    except:
        return None

# Recommendation Button
if st.button("🍴 Get Recommendations"):
    recommendations = recommender.get_recommendations_for_user(selected_author, top_n)

    if recommendations:
        st.success(f"🌟 Top {top_n} AI Picks for {selected_author}")
        cols = st.columns(3)

        for idx, title in enumerate(recommendations):
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
    else:
        st.warning("No recommendations found for this user.")
