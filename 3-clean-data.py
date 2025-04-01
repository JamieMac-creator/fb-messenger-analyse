import json
from datetime import datetime
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import os


def process_dir(input_dir):
    print("Processing directory: " + input_dir)
    js={}
    valid_dir=True
    for index, file_name in enumerate(os.listdir(input_dir)):
        if os.path.isdir(os.path.join(input_dir, file_name)):
            # Recursively process subdirectories
            valid_dir = False
            process_dir(os.path.join(input_dir, file_name))
        else:
            file_path = os.path.join(input_dir, file_name)
            print("Processing file: " + file_path)
            with open(file_path, mode="r", encoding="utf-8") as read_file:
                json_i = json.load(read_file, strict=False)
            if js == {}:
                js = json_i
            else:
                print("PROCESSING next file")
                js["messages"] = js["messages"] + json_i["messages"]
    if valid_dir:
        for message in js["messages"]:
            if 'is_geoblocked_for_viewer' in message:
                del message['is_geoblocked_for_viewer']
            if 'is_unsent_image_by_messenger_kid_parent' in message:
                del message['is_unsent_image_by_messenger_kid_parent']

        output_file = "clean_" + os.path.basename(input_dir) + ".json"
        output_dir = os.path.join("cleaned", input_dir)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, output_file), 'w', encoding='utf-8') as f:
            json.dump(js, f, ensure_ascii=False, indent=4)
        
process_dir('data/inbox')