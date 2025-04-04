import pandas as pd
import os

class RecipePreprocessor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def load_data(self):
        print(f"📥 Loading data from {self.input_path}")
        self.data = pd.read_csv(self.input_path)

    def clean_data(self):
        print("🧹 Cleaning data (removing rows with empty values)...")
        self.data.dropna(inplace=True)

    def preprocess_ingredients(self):
        print("🔪 Preprocessing ingredients (lowercase + separator cleanup)...")
        self.data['ingredients'] = self.data['ingredients'].apply(lambda x: " | ".join(x.lower().split('|')))

    def preprocess_instructions(self):
        print("📖 Preprocessing instructions (strip + capitalize)...")
        self.data['instructions'] = self.data['instructions'].apply(lambda x: x.strip().capitalize())

    def save_cleaned_data(self):
        print(f"💾 Saving cleaned data to {self.output_path}")
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.data.to_csv(self.output_path, index=False)

    def run_pipeline(self):
        self.load_data()
        self.clean_data()
        self.preprocess_ingredients()
        self.preprocess_instructions()
        self.save_cleaned_data()


if __name__ == "__main__":
    # Define paths relative to the project root
    input_csv = "data/recipes_final_dataset.csv"
    output_csv = "data/recipes_cleaned.csv"

    preprocessor = RecipePreprocessor(input_path=input_csv, output_path=output_csv)
    preprocessor.run_pipeline()
