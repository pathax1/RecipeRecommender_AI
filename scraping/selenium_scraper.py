import csv
import time
import random
import requests
import os
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class RecipeURLScraper:
    def __init__(self, driver_path, category_csv):
        options = Options()
        options.add_experimental_option("detach", True)
        self.driver = webdriver.Chrome(service=Service(driver_path), options=options)
        self.category_csv = category_csv
        self.recipe_links = []
        self.subcategory_data = []
        self.final_output_path = r"C:\\Users\\Autom\\PycharmProjects\\RecipeRecommender_AI\\data\\recipes_final_dataset.csv"
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:118.0) Gecko/20100101 Firefox/118.0",
            "Mozilla/5.0 (Linux; Android 13; SM-G998U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Mobile Safari/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 16_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.3 Mobile/15E148 Safari/604.1"
        ]

    def scroll_to_load_more(self, scroll_times=10):
        for _ in range(scroll_times):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

    def extract_subcategory_recipes(self):
        print("\n📦 Starting subcategory recipe name + URL extraction...")
        with open(r"C:\\Users\\Autom\\PycharmProjects\\RecipeRecommender_AI\\data\\recipe_urls.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                url = row["recipe_url"]
                print(f"[{idx+1}] Opening subcategory: {url}")
                self.driver.get(url)
                try:
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "mntl-sc-list-item"))
                    )
                    self.scroll_to_load_more(scroll_times=8)

                    blocks = self.driver.find_elements(By.CLASS_NAME, "mntl-sc-list-item")
                    print(f"  ↳ Found {len(blocks)} recipe items")

                    for block in blocks:
                        try:
                            title_element = block.find_element(By.CLASS_NAME, "mntl-sc-block-heading__text")
                            recipe_name = title_element.text.strip()
                            link_element = block.find_element(By.CLASS_NAME, "mntl-sc-block-universal-featured-link__link")
                            recipe_url = link_element.get_attribute("href")
                            self.subcategory_data.append({"recipe": recipe_name, "url": recipe_url})
                        except:
                            continue
                except Exception as e:
                    print(f"⚠️ Skipped {url}: {e}")

    def extract_final_recipe_details(self, max_recipes=100000, batch_start=0):
        print("\n🧠 Extracting full recipe data using BeautifulSoup...")
        actual_count = min(max_recipes, len(self.subcategory_data) - batch_start)
        print(f"🔍 Extracting {actual_count} recipes (Batch: {batch_start} to {batch_start + actual_count})")

        with open(self.final_output_path, "a", encoding="utf-8", newline="") as f:
            writer = None
            for idx, item in enumerate(self.subcategory_data[batch_start:batch_start + max_recipes]):
                url = item["url"]
                print(f"[{idx + 1 + batch_start}] Fetching: {url}")
                try:
                    headers = {
                        "User-Agent": random.choice(self.user_agents),
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                        "Accept-Encoding": "gzip, deflate, br",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Connection": "keep-alive",
                        "DNT": "1",
                        "Referer": "https://www.google.com/search?q=easy+chicken+recipes",
                        "Upgrade-Insecure-Requests": "1",
                        "Sec-Fetch-Dest": "document",
                        "Sec-Fetch-Mode": "navigate",
                        "Sec-Fetch-Site": "none",
                        "Sec-Fetch-User": "?1"
                    }

                    response = requests.get(url, headers=headers, timeout=15)
                    soup = BeautifulSoup(response.content, 'html.parser')

                    title = soup.find("h1", class_="article-heading text-headline-400")
                    title_text = title.get_text(strip=True) if title else ""

                    ingredients = [li.get_text(strip=True) for li in soup.select("ul.mm-recipes-structured-ingredients__list li")]
                    instructions = [li.get_text(strip=True) for li in soup.select("div.mm-recipes-steps__content ol li") if li.get_text(strip=True)]

                    rating = soup.find("div", class_="mm-recipes-review-bar__rating")
                    rating_val = rating.get_text(strip=True) if rating else ""

                    author_span = soup.find("span", class_="mntl-attribution__item-name")
                    author_name = author_span.get_text(strip=True) if author_span else ""

                    meta_block = soup.select("div.mm-recipes-details__item")
                    meta_dict = {}
                    for div in meta_block:
                        label = div.find("div", class_="mm-recipes-details__label")
                        value = div.find("div", class_="mm-recipes-details__value")
                        if label and value:
                            key = label.get_text(strip=True).replace(":", "")
                            meta_dict[key] = value.get_text(strip=True)

                    prep_time = meta_dict.get("Prep Time", "")
                    cook_time = meta_dict.get("Cook Time", "")
                    total_time = meta_dict.get("Total Time", "")
                    servings = meta_dict.get("Servings", "")

                    image = soup.find("img", class_="primary-image__image")
                    image_url = image["src"] if image else ""

                    item_dict = {
                        "title": title_text,
                        "url": url,
                        "rating": rating_val,
                        "author": author_name,
                        "ingredients": " | ".join(ingredients),
                        "instructions": " | ".join(instructions),
                        "prep_time": prep_time,
                        "cook_time": cook_time,
                        "total_time": total_time,
                        "servings": servings,
                        "image_url": image_url
                    }

                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=item_dict.keys())
                        if not os.path.exists(self.final_output_path) or os.path.getsize(self.final_output_path) == 0:
                            writer.writeheader()
                    writer.writerow(item_dict)

                    time.sleep(random.uniform(4.0, 7.0))

                except Exception as e:
                    print(f"⚠️ Failed to fetch from {url}: {e}")

    def save_subcategory_recipes(self):
        output_path = r"C:\\Users\\Autom\\PycharmProjects\\RecipeRecommender_AI\\data\\subcategory_recipes.csv"
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["recipe", "url"])
            writer.writeheader()
            for row in self.subcategory_data:
                writer.writerow(row)
        print(f"✅ Saved {len(self.subcategory_data)} subcategory recipes to: {output_path}")

if __name__ == "__main__":
    DRIVER_PATH = r"C:\\Users\\Autom\\PycharmProjects\\RecipeRecommender_AI\\chromedriver.exe"
    CATEGORY_CSV_PATH = r"C:\\Users\\Autom\\PycharmProjects\\RecipeRecommender_AI\\data\\recipes_dataset.csv"

    scraper = RecipeURLScraper(driver_path=DRIVER_PATH, category_csv=CATEGORY_CSV_PATH)
    scraper.extract_subcategory_recipes()
    scraper.save_subcategory_recipes()

    # Load saved subcategories
    sub_df = pd.read_csv(r"C:\\Users\\Autom\\PycharmProjects\\RecipeRecommender_AI\\data\\subcategory_recipes.csv")
    scraper.subcategory_data = sub_df.to_dict("records")

    total_rows = len(scraper.subcategory_data)
    batch_size = 100

    for start in range(0, total_rows, batch_size):
        scraper.extract_final_recipe_details(max_recipes=batch_size, batch_start=start)

    print("✅ All batches processed!")
