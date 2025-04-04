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
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--window-size=1920,1080')
        options.add_argument("--log-level=3")
        options.add_argument(f'user-agent={random.choice([
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:118.0) Gecko/20100101 Firefox/118.0",
            "Mozilla/5.0 (Linux; Android 13; SM-G998U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Mobile Safari/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 16_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.3 Mobile/15E148 Safari/604.1"
        ])}')

        self.driver = webdriver.Chrome(service=Service(driver_path), options=options)
        self.category_csv = category_csv
        self.final_output_path = r"C:\\Users\\Autom\\PycharmProjects\\RecipeRecommender_AI\\data\\recipes_final_dataset.csv"
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:118.0) Gecko/20100101 Firefox/118.0",
            "Mozilla/5.0 (Linux; Android 13; SM-G998U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Mobile Safari/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 16_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.3 Mobile/15E148 Safari/604.1"
        ]

    def extract_subcategory_recipes(self):
        print("\n📦 Starting subcategory recipe name + URL extraction...")
        self.subcategory_data_path = r"C:\\Users\\Autom\\PycharmProjects\\RecipeRecommender_AI\\data\\subcategory_recipes.csv"
        with open(r"C:\\Users\\Autom\\PycharmProjects\\RecipeRecommender_AI\\data\\recipe_urls.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header_written = False
            for idx, row in enumerate(reader):
                url = row["recipe_url"]
                print(f"[{idx+1}] Opening subcategory: {url}")
                self.driver.get(url)
                try:
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "mntl-sc-list-item"))
                    )
                    for _ in range(8):
                        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                        time.sleep(2)

                    blocks = self.driver.find_elements(By.CLASS_NAME, "mntl-sc-list-item")
                    print(f"  ↳ Found {len(blocks)} recipe items")

                    with open(self.subcategory_data_path, "a", encoding="utf-8", newline="") as out_f:
                        writer = csv.DictWriter(out_f, fieldnames=["recipe", "url"])
                        if not header_written and (not os.path.exists(self.subcategory_data_path) or os.path.getsize(self.subcategory_data_path) == 0):
                            writer.writeheader()
                            header_written = True
                        for block in blocks:
                            try:
                                title_element = block.find_element(By.CLASS_NAME, "mntl-sc-block-heading__text")
                                recipe_name = title_element.text.strip()
                                link_element = block.find_element(By.CLASS_NAME, "mntl-sc-block-universal-featured-link__link")
                                recipe_url = link_element.get_attribute("href")
                                writer.writerow({"recipe": recipe_name, "url": recipe_url})
                                out_f.flush()
                            except:
                                continue
                except Exception as e:
                    print(f"⚠️ Skipped {url}: {e}")

    def extract_final_recipe_details(self, batch_size=50):
        print("\n🧠 Extracting full recipe data using BeautifulSoup...")
        subcategory_path = r"C:\\Users\\Autom\\PycharmProjects\\RecipeRecommender_AI\\data\\subcategory_recipes.csv"
        with open(subcategory_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            total = len(reader)

        for start in range(0, total, batch_size):
            batch = reader[start:start+batch_size]
            print(f"🔍 Extracting Batch: {start} to {start+len(batch)}")
            for idx, item in enumerate(batch):
                url = item["url"]
                print(f"[{start + idx + 1}] Fetching: {url}")
                try:
                    headers = {
                        "User-Agent": random.choice(self.user_agents),
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                        "Accept-Encoding": "gzip, deflate, br",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Connection": "keep-alive",
                        "Referer": "https://www.google.com/",
                        "Upgrade-Insecure-Requests": "1"
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

                    file_exists = os.path.isfile(self.final_output_path)
                    with open(self.final_output_path, "a", encoding="utf-8", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=item_dict.keys())
                        if not file_exists or os.path.getsize(self.final_output_path) == 0:
                            writer.writeheader()
                        writer.writerow(item_dict)
                        f.flush()

                    print(f"✅ Saved: {title_text}")
                    del soup, response
                    time.sleep(random.uniform(4.0, 7.0))

                except Exception as e:
                    print(f"⚠️ Failed to fetch from {url}: {e}")

if __name__ == "__main__":
    DRIVER_PATH = r"C:\\Users\\Autom\\PycharmProjects\\RecipeRecommender_AI\\chromedriver.exe"
    CATEGORY_CSV_PATH = r"C:\\Users\\Autom\\PycharmProjects\\RecipeRecommender_AI\\data\\recipes_dataset.csv"

    scraper = RecipeURLScraper(driver_path=DRIVER_PATH, category_csv=CATEGORY_CSV_PATH)
    scraper.extract_subcategory_recipes()
    scraper.extract_final_recipe_details(batch_size=50)

    print("✅ All batches processed and saved!")
