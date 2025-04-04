import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from core.model_pipeline import NCFModel




class RecipeRecommender:
    def __init__(self, model_path, data_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_path = data_path
        self.model_path = model_path

        self.data = pd.read_csv(self.data_path)
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self._prepare_data()

        self.model = NCFModel(self.num_users, self.num_items)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def _prepare_data(self):
        self.data = self.data.drop_duplicates(subset=["title", "author"])
        self.data = self.data[["title", "author", "rating"]].dropna()

        self.data["user"] = self.user_encoder.fit_transform(self.data["author"])
        self.data["item"] = self.item_encoder.fit_transform(self.data["title"])

        self.num_users = self.data["user"].nunique()
        self.num_items = self.data["item"].nunique()

    def get_recommendations_for_user(self, author_name, top_n=5):
        if author_name not in self.user_encoder.classes_:
            print("Author not found in training data.")
            return []

        user_id = self.user_encoder.transform([author_name])[0]
        all_items = set(range(self.num_items))
        seen_items = set(self.data[self.data["user"] == user_id]["item"])

        unseen_items = list(all_items - seen_items)

        user_tensor = torch.LongTensor([user_id] * len(unseen_items)).to(self.device)
        item_tensor = torch.LongTensor(unseen_items).to(self.device)

        with torch.no_grad():
            scores = self.model(user_tensor, item_tensor)

        top_indices = torch.topk(scores, top_n).indices.cpu().numpy()
        top_item_ids = [unseen_items[i] for i in top_indices]
        top_titles = self.item_encoder.inverse_transform(top_item_ids)

        return list(top_titles)


if __name__ == "__main__":
    model_path = r"C:\Users\Autom\PycharmProjects\RecipeRecommender_AI\data\ncf_model.pth"
    data_path = "data/recipes_cleaned.csv"

    recommender = RecipeRecommender(model_path=model_path, data_path=data_path)

    author = input("Enter author name (from dataset): ")
    top_n = int(input("How many recommendations do you want? "))

    recommendations = recommender.get_recommendations_for_user(author, top_n)
    print("\nRecommended Recipes:")
    for idx, title in enumerate(recommendations, 1):
        print(f"{idx}. {title}")
