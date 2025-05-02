import requests
import json

json_data = {"results": []}
index = 0
for j in ['1']:
    url = 'https://tcgcsv.com/tcgplayer/' + j + '/groups'
    response = requests.get(url)

    data = response.json()
    data = data['results']
    for i in data:
        id = i['groupId']
        url1 = 'https://tcgcsv.com/tcgplayer/' + j + '/' + str(id) + '/prices'
        response1 = requests.get(url1)
        data1 = response1.json()
        data1 = data1['results']
        # result = result + str(data1)
        for k in data1:
            index = index + 1
            json_data['results'].append(k)
with open('magic_card_price_data.json', 'w', encoding='utf-8') as file:
    json.dump(json_data, file, indent=4)