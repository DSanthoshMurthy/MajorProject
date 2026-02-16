import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import time
import os

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with single space
    text = text.strip()
    return text

def scrape_url(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # generic scraping strategy: get all paragraph text
            # Adjust this strictly if you know the specific HTML class of your news source
            paragraphs = soup.find_all('p')
            full_text = ' '.join([p.get_text() for p in paragraphs])
            
            return clean_text(full_text)
        else:
            return None
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None



INPUT_CSV = "data/raw_data.csv"
OUTPUT_CSV = "data/finbert_training_data.csv"
BATCH_SIZE = 100
MAX_ROWS = 5000

if not os.path.exists(INPUT_CSV):
    print(f"Error: {INPUT_CSV} not found.")
    exit()

df = pd.read_csv(INPUT_CSV)
df_subset = df.head(MAX_ROWS).copy()
total_rows = len(df_subset)

df_subset = df.head(MAX_ROWS).copy()

if os.path.exists(OUTPUT_CSV):
    os.remove(OUTPUT_CSV)
    print(f"Deleted old {OUTPUT_CSV} to start fresh.")


for start_index in range(0, total_rows, BATCH_SIZE):
    end_index = min(start_index + BATCH_SIZE, total_rows)
    current_batch = df_subset.iloc[start_index:end_index].copy()
    
    print(f"Processing rows {start_index} to {end_index}...")
    
    # Scrape URLs for this batch
    batch_texts = []
    for idx, row in current_batch.iterrows():
        text = scrape_url(row['URL'])
        batch_texts.append(text)
        time.sleep(0.2) # Small delay to be polite to servers
    
    current_batch['text'] = batch_texts
    
    current_batch = current_batch.dropna(subset=['text'])
    current_batch = current_batch[current_batch['text'] != ""]
    
    if 'URL' in current_batch.columns:
        current_batch = current_batch.drop(columns=['URL'])
    
    
    write_header = (start_index == 0)
    current_batch.to_csv(OUTPUT_CSV, mode='a', header=write_header, index=False)
    
    print(f"  -> Saved {len(current_batch)} valid rows to {OUTPUT_CSV}")

print("------------------------------------------------")
print(f"Scraping Complete! Data saved to {OUTPUT_CSV}")