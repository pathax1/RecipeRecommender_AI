import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from math import sqrt

class NCFModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=50):
        super(NCFModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        concat = torch.cat([user_embeds, item_embeds], dim=-1)
        output = self.fc_layers(concat)
        return output.squeeze()


class RecipeModelPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

    def load_and_prepare_data(self):
        df = pd.read_csv(self.data_path)
        df = df.drop_duplicates(subset=["title", "author"])
        df = df[["title", "author", "rating"]].dropna()

        df["user"] = self.user_encoder.fit_transform(df["author"])
        df["item"] = self.item_encoder.fit_transform(df["title"])
        df["rating"] = df["rating"] / 5.0

        self.num_users = df["user"].nunique()
        self.num_items = df["item"].nunique()

        train, test = train_test_split(df, test_size=0.2, random_state=42)
        return train, test, df

    def train_ncf_model(self, train_data, epochs=10, lr=0.001):
        model = NCFModel(self.num_users, self.num_items).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            model.train()
            users = torch.LongTensor(train_data["user"].values).to(self.device)
            items = torch.LongTensor(train_data["item"].values).to(self.device)
            ratings = torch.FloatTensor(train_data["rating"].values).to(self.device)

            optimizer.zero_grad()
            outputs = model(users, items)
            loss = loss_fn(outputs, ratings)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

        self.model = model
        torch.save(model.state_dict(), r"data/ncf_model.pth")
        print("Model saved as ncf_model.pth")

    def evaluate_ncf_model(self, test_data):
        self.model.eval()
        users = torch.LongTensor(test_data["user"].values).to(self.device)
        items = torch.LongTensor(test_data["item"].values).to(self.device)
        ratings = torch.FloatTensor(test_data["rating"].values).to(self.device)

        with torch.no_grad():
            predictions = self.model(users, items)
            mse = nn.MSELoss()(predictions, ratings).item()
            rmse = np.sqrt(mse)
            print(f"NCF RMSE: {rmse:.4f}")
        return rmse

    def run_content_based(self, df):
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['title'])
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        def get_rmse_for_user(user_id):
            user_data = df[df["author"] == user_id]
            predictions = []
            actuals = []
            for _, row in user_data.iterrows():
                idx = df[df['title'] == row['title']].index[0]
                sim_scores = list(enumerate(similarity_matrix[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                for i, _ in sim_scores[1:]:
                    if df.iloc[i]['title'] != row['title']:
                        predictions.append(df.iloc[i]['rating'])
                        actuals.append(row['rating'])
                        break
            if not predictions:
                return None
            return round(sqrt(mean_squared_error(actuals, predictions)), 4)

        sample_users = df['author'].dropna().unique()[:10]
        scores = [get_rmse_for_user(user) for user in sample_users if get_rmse_for_user(user) is not None]
        avg_rmse = round(np.mean(scores), 4) if scores else None
        print(f"Content-Based RMSE: {avg_rmse}")
        return avg_rmse

    def run_collaborative_filtering(self, df):
        pivot_table = df.pivot_table(index='author', columns='title', values='rating')
        pivot_table.fillna(0, inplace=True)
        similarity_matrix = cosine_similarity(pivot_table)
        np.fill_diagonal(similarity_matrix, 0)

        predictions = []
        actuals = []

        for idx, author in enumerate(pivot_table.index):
            similar_users = similarity_matrix[idx]
            top_user_idx = similar_users.argsort()[-1]

            user_ratings = pivot_table.iloc[top_user_idx]
            top_recipe_idx = user_ratings[user_ratings > 0].idxmax()

            if not pd.isna(top_recipe_idx):
                predictions.append(user_ratings[top_recipe_idx])
                actual = pivot_table.loc[author][top_recipe_idx]
                actuals.append(actual)

        rmse = round(sqrt(mean_squared_error(actuals, predictions)), 4) if predictions else None
        print(f"Collaborative Filtering RMSE: {rmse}")
        return rmse

    def run_pipeline(self):
        train_data, test_data, full_df = self.load_and_prepare_data()
        self.train_ncf_model(train_data)
        ncf_rmse = self.evaluate_ncf_model(test_data)
        cb_rmse = self.run_content_based(full_df)
        cf_rmse = self.run_collaborative_filtering(full_df)
        return ncf_rmse, cb_rmse, cf_rmse


if __name__ == "__main__":
    data_file = "data/recipes_cleaned.csv"
    pipeline = RecipeModelPipeline(data_path=data_file)
    ncf_rmse, cb_rmse, cf_rmse = pipeline.run_pipeline()
    print(f"\nModel Comparison:\nNCF RMSE: {ncf_rmse}\nContent-Based RMSE: {cb_rmse}\nCollaborative Filtering RMSE: {cf_rmse}")