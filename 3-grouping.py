import json
from datetime import datetime
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import os

js={}

for index, file_name in enumerate(os.listdir('cleaned')):
    file_path = os.path.join('cleaned', file_name)
    print(file_name+":")
    with open(file_path, mode="r", encoding="utf-8") as read_file:
        json_i = json.load(read_file, strict=False)
    if js == {}:
        print("processing first file")
        js = json_i
    else:
        print("processing next file")
        js["messages"] = js["messages"] + json_i["messages"]

for message in js["messages"]:
    if 'is_geoblocked_for_viewer' in message:
        del message['is_geoblocked_for_viewer']
    if 'is_unsent_image_by_messenger_kid_parent' in message:
        del message['is_unsent_image_by_messenger_kid_parent']

with open('cleaned_messages.json', 'w') as f:
    json.dump(js, f, ensure_ascii=False, indent=4)