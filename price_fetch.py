import requests
import json
url = 'https://www.tcgplayer.com/product/232517/pokemon-shining-fates-shiny-vault-eternatus-v'
headers = {
    'accept': '*/*',
    'referer': 'https://www.tcgplayer.com/',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36',
}

product_number = url.split('/')[4]

response = requests.get(f'https://mp-search-api.tcgplayer.com/v2/product/{product_number}/details?mpfev=3582', headers=headers)

data = json.loads(response.content)
print(f"MarketPrice is {data['marketPrice']}")
print(f"LowestPrice is {data['lowestPrice']}")
print(f"MedianPrice is {data['medianPrice']}")
print(f"all data is {data}")
with open('data_updated.json', 'w') as json_file:
    json.dump(json.loads(response.content), json_file, indent=4)