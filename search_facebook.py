# scraper.py
import asyncio
from pyppeteer import launch
from urllib.parse import quote
import re

browser = None
async def get_href(query):
    async def goto_more(page):
        await page.waitForSelector('.bi.bj')
        href = await page.evaluate('''() => {
            const element = document.querySelector('.bi.bj a');
            return element ? element.getAttribute('href') : null;
        }''')
        await asyncio.sleep(0.5)
        return href

    async def extract_text(page):
        await page.waitForSelector('.z.ba')
        await page.waitForSelector('.be.bf')
        texts = await page.evaluate('''() => {
            const elements = document.querySelectorAll('.z.ba');
            const timestamps = document.querySelectorAll('.be.bf');
            return Array.from(elements).map((element, index) => {
                const text = element.textContent.trim() + '\\n';
                const timestamp = timestamps[index].textContent.trim();
                return { text};
            });
        }''')

        pattern = r'\d+\s*(?:hrs|min|hr)s?'

        for text in texts:
            #print('Text:', text['text'])
            times = re.findall(pattern, text['text'])
            #print(times)

    async def extract_text_2(page):
        await page.waitForSelector('.ca.cb')
        texts = await page.evaluate('''() => {
            const elements = document.querySelectorAll('.ca.cb');
            return Array.from(elements).map(element => element.textContent.trim());
        }''')
        date_pattern = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\s+at\s+\d{1,2}:\d{2}\s+[AP]M'
        duration_pattern = r'\d+\s*(?:hrs|min|hr)s?'
        result = ""
        for text in texts:
            #print(text, '\\n')
            dates = re.findall(date_pattern, text)
            durations = re.findall(duration_pattern, text)
            time = ""
            if dates:
                #print('Dates:', dates)
                time = dates
            if durations:
                #print('Time Durations:', durations[-1])
                time = durations[-1]
            
            result += f'{text}\nTime: {time}\n'
        return result

    async def goto_full_story(page):
        await page.waitForSelector('a')
        hrefs = await page.evaluate('''() => {
            const elements = document.querySelectorAll('a');
            return Array.from(elements)
                .map(element => element.getAttribute('href'))
                .filter(href => href && href.startsWith('/story'));
        }''')

        hrefs = list(set(hrefs))
        '''
        for href in hrefs:
            print('href:', href)
        '''
        return hrefs

    global browser
    browser = await launch(headless=False, userDataDir='./userdata')
    page = await browser.newPage()
    
    # Set custom user agent
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:131.0) Gecko/20100101 Firefox/131.0'
    await page.setUserAgent(user_agent)
    
    await page.setRequestInterception(True)
    page.on('request', lambda req: asyncio.ensure_future(req.abort()) if req.resourceType == 'image' else asyncio.ensure_future(req.continue_()))

    url = f'https://mbasic.facebook.com/search/posts?q={quote(query)}&filters=eyJyZWNlbnRfcG9zdHM6MCI6IntcIm5hbWVcIjpcInJlY2VudF9wb3N0c1wiLFwiYXJnc1wiOlwiXCJ9In0%3D'
    await page.goto(url)

    more = await goto_more(page)
    await page.goto(more)
    
    await page.setRequestInterception(False)
    
    more = await goto_more(page)
    await page.goto(more)

    results = await extract_text_2(page)
    full_story = await goto_full_story(page)
    #print("Gotten")

    await browser.close()
    #print(results)
    return results

    results = []
    for href in full_story:
        await asyncio.sleep(0.5)
        try:
            await page.goto('https://mbasic.facebook.com' + href)
            await extract_text(page)
        except Exception as e:
            print(e)
    return results

if __name__ == "__main__":
    import sys
    query = sys.argv[1]# if len(sys.argv) > 1 else "World"
    print(asyncio.get_event_loop().run_until_complete(get_href(query)))
    
