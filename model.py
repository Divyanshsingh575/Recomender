import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('merge_data.csv')

class AuctionRecommender:
    def __init__(self, df):
        self.df = df.copy()
        self.clean_data()
        self.create_matrices()

    def clean_data(self):
        self.df[['auction_price', 'min_price', 'bid_price', 'user_id']] = self.df[
            ['auction_price', 'min_price', 'bid_price', 'user_id']].apply(pd.to_numeric, errors='coerce')
        self.df.fillna({'category_name': 'Unknown', 'description': '', 'auction_name': ''}, inplace=True)
        self.df['item_features'] = self.df[['auction_name', 'category_name', 'description']].agg(' '.join, axis=1)

    def create_matrices(self):
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
        self.item_matrix = self.tfidf.fit_transform(self.df['item_features'])
        self.item_similarity = cosine_similarity(self.item_matrix)
        self.user_item_matrix = self.df.pivot_table(values='bid_price', index='user_id', columns='auction_name', fill_value=0)
    
    def content_based(self, item_idx, n=5):
        return self.df.iloc[self.item_similarity[item_idx].argsort()[::-1][1:n+1]]
    
    def collaborative(self, user_id, n=5):
        if user_id not in self.user_item_matrix.index:
            return pd.DataFrame()
        sim_users = cosine_similarity([self.user_item_matrix.loc[user_id]], self.user_item_matrix)[0].argsort()[::-1][1:6]
        return self.df[self.df['user_id'].isin(self.user_item_matrix.index[sim_users])].head(n)

    def recommend(self, user_id, prev_bid_price, preferred_category, n=3):
        df_filtered = self.df.copy()
        df_filtered['score'] = 0
        
        user_items = self.df[self.df['user_id'] == user_id]
        if not user_items.empty:
            df_filtered.loc[user_items.index[-1], 'score'] += 3
        
        df_filtered.loc[self.collaborative(user_id).index, 'score'] += 3
        df_filtered.loc[df_filtered['category_name'].str.lower() == preferred_category.lower(), 'score'] += 2
        
        related = {'Automotive': ['Cars'], 'Cars': ['Automotive'], 'Real Estate': ['Rental Spaces']}
        if preferred_category in related:
            df_filtered.loc[df_filtered['category_name'].isin(related[preferred_category]), 'score'] += 1
        
        df_filtered['score'] += (1 - abs(df_filtered['auction_price'] - prev_bid_price) / prev_bid_price).clip(0, 1) * 2
        
        recs = df_filtered[(df_filtered['auction_price'].between(prev_bid_price * 0.6, prev_bid_price * 1.8))]
        recs = recs.nlargest(n, 'score')[['auction_name', 'category_name', 'auction_price', 'min_price', 'description', 'score']]
        return recs if not recs.empty else df_filtered.nlargest(n, 'score')

    def save_model(self, filepath="auction_recommender.pkl"):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath="auction_recommender.pkl"):
        with open(filepath, "rb") as f:
            return pickle.load(f)

def get_recommendations(data_df, user_id, prev_bid_price, preferred_category):
    try:
        recommender = AuctionRecommender(data_df)
        recommender.save_model()  
        return recommender.recommend(user_id, prev_bid_price, preferred_category)
    except Exception as e:
        return f"Error: {str(e)}"


recommender = AuctionRecommender(df)
recommender.save_model("auction_recommender.pkl")



# recommendations = get_recommendations(
#     df,
#     user_id=3,
#     prev_bid_price=170,
#     preferred_category="Automotive"
# )

# print(recommendations)
