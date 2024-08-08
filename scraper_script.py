# scraper_script.py
import asyncio
from scraper import get_href
import sys
import json

async def main(query):
    result = await get_href(query)
    print(json.dumps(result))  # Print result as JSON

if __name__ == "__main__":
    query = sys.argv[1]  # Get query from command-line arguments
    asyncio.run(main(query))
