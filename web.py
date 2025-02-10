import streamlit as st
import pandas as pd
import pickle
from model import AuctionRecommender

# Load the model form the given file path
@st.cache_resource
def load_model(filepath="auction_recommender.pkl"):
    with open(filepath, "rb") as f:
        return pickle.load(f)

# Load model
model = load_model("auction_recommender.pkl") 
st.title("Auction Recommender System")

st.sidebar.header("Enter Details")

user_id = st.sidebar.number_input("User ID", min_value=1, step=1)
prev_bid_price = st.sidebar.number_input("Previous Bid Price", min_value=0, step=20,max_value=180)
preferred_category = st.sidebar.text_input("Preferred Category", "")

if st.sidebar.button("Get Recommendations"):
    if user_id and prev_bid_price and preferred_category:
        # Get recommendations
        recommendations = model.recommend(user_id, prev_bid_price, preferred_category)
        
        if recommendations.empty:
            st.warning("No recommendations found. Try adjusting the inputs.")
        else:
            st.subheader("🔹 Recommended Auctions for You:")
            st.dataframe(recommendations)
    else:
        st.error("Please fill in all details before proceeding.")

# Add Chatbot in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Chat with our AI Bot 🤖")
st.sidebar.markdown(
    '<iframe src="https://console.dialogflow.com/api-client/demo/embedded/a0c510cf-e130-4cfd-8447-6bacecaaeb56" '
    'width="220" height="430"></iframe>',
    unsafe_allow_html=True
)

st.markdown("Developed by **Divyansh Singh** ")
