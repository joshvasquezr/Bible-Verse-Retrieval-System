import requests

url = "https://openbible.com/textfiles/kjv.txt"
response = requests.get(url)
response.raise_for_status()  # raises error if download failed

with open("data/kjv.txt", "w", encoding="utf-8") as f:
    f.write(response.text)

print("Bible downloaded as kjv.txt")