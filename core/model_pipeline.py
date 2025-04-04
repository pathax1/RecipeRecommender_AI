import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
        df = df[["title", "author", "rating"]]
        df = df.dropna()

        df["user"] = self.user_encoder.fit_transform(df["author"])
        df["item"] = self.item_encoder.fit_transform(df["title"])
        df["rating"] = df["rating"] / 5.0  # normalize between 0 and 1

        self.num_users = df["user"].nunique()
        self.num_items = df["item"].nunique()

        train, test = train_test_split(df, test_size=0.2, random_state=42)
        return train, test

    def train_model(self, train_data, epochs=10, lr=0.001):
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

        # Save the trained model
        torch.save(model.state_dict(), r"C:\Users\Autom\PycharmProjects\RecipeRecommender_AI\data\ncf_model.pth")
        print("Model saved as ncf_model.pth")

    def evaluate_model(self, test_data):
        self.model.eval()
        users = torch.LongTensor(test_data["user"].values).to(self.device)
        items = torch.LongTensor(test_data["item"].values).to(self.device)
        ratings = torch.FloatTensor(test_data["rating"].values).to(self.device)

        with torch.no_grad():
            predictions = self.model(users, items)
            mse = nn.MSELoss()(predictions, ratings).item()
            rmse = np.sqrt(mse)
            print(f"Test RMSE: {rmse:.4f}")

    def run_pipeline(self):
        train_data, test_data = self.load_and_prepare_data()
        self.train_model(train_data)
        self.evaluate_model(test_data)


if __name__ == "__main__":
    data_file = "data/recipes_cleaned.csv"
    pipeline = RecipeModelPipeline(data_path=data_file)
    pipeline.run_pipeline()