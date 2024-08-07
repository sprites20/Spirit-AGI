import requests

def search(query):
    server_url = 'http://127.0.0.1:5000/search'
    response = requests.get(server_url, params={'query': query})
    
    if response.status_code == 200:
        results = response.json()
        print("Articles:")
        for article in results.get('articles', []):
            print(f"Dates: {article.get('dates', [])} | Durations: {article.get('durations', [])}")
        
        print("\nFull Stories:")
        for href in results.get('full_story', []):
            print(f"https://mbasic.facebook.com{href}")
    else:
        print("Error fetching results from server")

if __name__ == '__main__':
    query = input("Enter your search query: ")
    search(query)