import asyncio
from pyppeteer import launch
from bs4 import BeautifulSoup
import requests
import os
import pprint
import re
from datetime import datetime, timedelta
import json

import time
from openai import OpenAI

async def search_google(search_query):
    browser = await launch(headless=True)
    page = await browser.newPage()
    await page.goto('https://www.google.com')

    # Perform the search
    await page.type('input[name=q]', search_query)
    await page.keyboard.press('Enter')
    await page.waitForNavigation()

    # Extract the search results' HTML
    search_results = await page.content()
    await browser.close()

    return search_results

def extract_links(html):
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    for index, link_element in enumerate(soup.find_all('a', href=True)):
        link = link_element['href']
        if link.startswith('/url?'):
            link = link.split('url=')[1].split('&')[0]
        
        # Extract additional attributes and context
        link_text = link_element.get_text(strip=True)
        title = link_element.get('title', '')
        data_attributes = {attr: link_element.get(attr) for attr in link_element.attrs if attr.startswith('data-')}
        class_name = ' '.join(link_element.get('class', []))
        parent_div = link_element.find_parent('div').get('class') if link_element.find_parent('div') else None
        grandparent_div = link_element.find_parent('div').find_parent('div').get('class') if link_element.find_parent('div') and link_element.find_parent('div').find_parent('div') else None
        
        data = {
            'href': link,
            'text': link_text,
            'title': title,
            'data_attributes': data_attributes,
            'class': class_name,
            'parent_div_class': parent_div,
            'grandparent_div_class': grandparent_div,
            'position': index
        }
        
        #pprint.pp(data)
        links.append(data)
        
    return links
        
def sanitize_filename(filename):
    # Replace invalid characters with underscores
    return re.sub(r'[\\/*?:"<>|]', '_', filename)
        
def send_one_doc_to_vectara(json_data, metadata_args, file_path):
    metadata = {
        #"metadata_key": "metadata_value",
        "date_downloaded": metadata_args["date_downloaded"],
        "date_uploaded": re.sub(r'[/:.]', '_', datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")),
        "epoch" : int(time.time()),
        "search_query": metadata_args["search_query"],
        "url" : metadata_args["url"],
        "filename" : metadata_args["filename"],
    }
    json_data_bytes = json.dumps(json_data).encode('utf-8')
    url = "https://api.vectara.io/v1/upload?c=4239971555&o=2"
    headers = {
        'x-api-key': 'zwt__LjU4zVE9TBF1tU4SbJ0rrjT1QWwI2sXXp4iGQ'
    }
    now = datetime.utcnow().strftime("%Y_%m_%d %H_%M_%S UTC")
    # Save the JSON-formatted string to a file
    with open(f"pages_json/{metadata['filename']}.json", "w") as file:
        file.write(json.dumps(json_data))
    files = {
        "file": (f"{metadata['filename']}", json_data_bytes, 'rb'),
        "doc_metadata": (None, json.dumps(metadata)),  # Replace with your metadata
    }
    response = requests.post(url, headers=headers, files=files)
    print(response.text)

async def download_webpage_text(url, folder_path, search_query):
    try:
        response = requests.get(url, timeout=3)
        response.raise_for_status()  # Raise HTTPError for bad responses
        # Extract text from HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        webpage_text = soup.get_text(separator='\n', strip=True)

        download_time = re.sub(r'[/:.]', '_', datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))
        # Generate a valid filename from the URL
        filename = sanitize_filename(download_time) + "__" + sanitize_filename(url.replace('http://', '').replace('https://', '').replace('/', '_')) + '.html'
        file_path = os.path.join(folder_path, filename)
        
        # Save the webpage text to a file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(response.text)
        text_path = f"text_ver/{search_query}/"
        if not os.path.exists(text_path):
            os.makedirs(text_path)
        # Save the webpage text to a file
        with open(os.path.join(text_path, filename), 'w', encoding='utf-8') as file:
            file.write(webpage_text)
            
        return file_path, filename, download_time, webpage_text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None
    
def generate_instructs(context):
    #user_prompt = random_search_query
    # Format UTC time
    print("Generating instructs")
    formatted_utc_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    TOGETHER_API_KEY = "5391e02b5fbffcdba1e637eada04919a5a1d9c9dfa5795eafe66b6d464d761ce"

    client = OpenAI(
    api_key=TOGETHER_API_KEY,
    base_url='https://api.together.xyz/v1',
    )
    chat_completion = client.chat.completions.create(
    messages=[
        {
          "role": "system",
          "content": "You are a LORA trainer, you generate synthetic data for instruct finetuning.",
        },
        {
          "role": "user",
          "content": f'''
{context}

Based on the context

Generate a json for fine-tuning instruct model with LORA using the following:
Enclose the json like:
"""json 
{{
    "input": <the context/input>
    "instruction": <instruction based on the input>
    "output": <the output>
}},
{{<more>
}},
"""
''',
        }
      ],
      model="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )

    result = chat_completion.choices[0].message.content
    
    print(result)
    
    return result

def get_search_query(user_prompt):
    #user_prompt = random_search_query
    # Format UTC time
    formatted_utc_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    TOGETHER_API_KEY = "5391e02b5fbffcdba1e637eada04919a5a1d9c9dfa5795eafe66b6d464d761ce"

    client = OpenAI(
    api_key=TOGETHER_API_KEY,
    base_url='https://api.together.xyz/v1',
    )
    chat_completion = client.chat.completions.create(
      messages=[
        {
          "role": "system",
          "content": "You want to search the web, and wants to Google about the topic. You answer in search engine query or multiple. Maximum is 1, EACH query is separated by brackets, {query1} ",
        },
        {
          "role": "user",
          "content": f"Date Now:{formatted_utc_time}\n{user_prompt}",
        }
      ],
      model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    )

    result = chat_completion.choices[0].message.content
    
    #print(result)
    # Regular expression to match text inside curly braces
    pattern = r"\{(.*?)\}"
    
    # Find all matches
    matches = re.findall(pattern, result)
    

    # Find all matches
    matches = re.findall(pattern, result)

    # Print the matches
    #matches.insert(0, user_prompt)
    print(matches)
    return matches

async def fetch_search_results(search_query, folder_path):
    search_results_html = await search_google(search_query)
    links = extract_links(search_results_html)
    valid_links = [link for link in links if link['class'] == "" and link['href'].startswith("http")]

    async def process_link(link):
        try:
            if link['parent_div_class'] and link['parent_div_class'][0] != 'Pg70bf':
                if link['grandparent_div_class']:
                    url = link['href']
                    print(link)
                    file_path = None
                    try:
                        date_downloaded = re.sub(r'[/:.]', '_', datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))
                        # Ensure download_webpage_text is async and returns a tuple
                        result = await download_webpage_text(url, folder_path, search_query)
                        
                        if result is None or not isinstance(result, tuple):
                            raise Exception("Failed to download the webpage text or unexpected return type.")
                        
                        file_path, filename, download_time, text = result
                        print(f"Downloaded file {url}")
                        date_uploaded = re.sub(r'[/:.]', '_', datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))
                        
                        metadata_args = {
                            "date_downloaded": date_downloaded,
                            "search_query": search_query,
                            "url": url,
                            "file_path": file_path,
                            "filename": filename,
                        }
                        temp_json = {
                            "search_query": search_query,
                            "date_downloaded": date_downloaded,
                            "date_uploaded": re.sub(r'[/:.]', '_', date_uploaded),
                            "query": None,
                            "url": url,
                            "text": text,
                        }
                        #print(text)
                        generate_instructs(text)
                        send_one_doc_to_vectara(temp_json, metadata_args, file_path)
                    except Exception as e:
                        print(f"Error fetching {url}: {e}")
                        return 0  # Return 0 on error
                    if file_path:
                        print(f"Downloaded and saved text from {url} to {file_path}")
                        return 1  # Return 1 on success
                    else:
                        print(f"Failed to download text from {url}")
                        return 0  # Return 0 on failure
        except Exception as e:
            print(f"Unexpected error with link processing: {e}")
        return 0  # Ensure that the function always returns an integer
    downloaded_count = 0
    max_its = 0
    
    while downloaded_count < 5 and max_its < 10:
        try:
            tasks = [process_link(link) for link in valid_links[:5-downloaded_count]]  # Process only the remaining links needed
            results = await asyncio.gather(*tasks)
            downloaded_count += sum(results)  # Add the number of successful downloads
            valid_links = valid_links[len(results):]  # Remove processed links from the list
            print(f"Total downloaded pages so far: {downloaded_count}")
        except Exception as e:
            print(f"Unexpected error in the download loop: {e}")
        max_its += 1
    
    if downloaded_count >= 5:
        print(f"Successfully downloaded 5 links.")
    else:
        print(f"Completed {max_its} iterations but only downloaded {downloaded_count} links.")

        
# Function to be called with the search query and folder path
def run_search_and_download(search_query):
    # Ensure the folder exists
    folder_path = f'downloaded_pages\\{search_query}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    asyncio.get_event_loop().run_until_complete(fetch_search_results(search_query, folder_path))


#get_search_query('search about pharmacology')
# Example usage
run_search_and_download('recent technology')