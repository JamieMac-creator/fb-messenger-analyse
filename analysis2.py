import json
from datetime import datetime
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np

with open("processed_messages.json", mode="r", encoding="utf-8") as read_file:
    messages = json.load(read_file, strict=False)

wordcloud = {}
for participant in messages["participants"]:
    wordcloud[participant["name"]] = []
# For each message dict in list of messages
for message in messages["messages"]:
    # Create human readable timestamp
    if 'timestamp_ms' in message:
        message['datetime'] = datetime.fromtimestamp(message['timestamp_ms'] / 1000).strftime('%Y-%m-%d %H:%M:%S')

    # Add words from message to list
    if 'content' in message and 'datetime' in message and datetime.strptime(message['datetime'], '%Y-%m-%d %H:%M:%S').year >= 2023:
        wordcloud[message["sender_name"]].extend(message['content'].split())
      


      


# Count lummy for each participant
# Count occurrences of "lummy" for each participant
lummy_counts = {participant: sum(1 for word in words if word.lower() == "lummy") for participant, words in wordcloud.items()}
word_counts = {participant: len(words) for participant, words in wordcloud.items()}
lummy_percentages = {participant: lummy_counts[participant] / word_counts[participant] for participant in lummy_counts}

# Create a separate bar chart for "lummy" counts
fig, ax = plt.subplots(3, dpi=300, figsize=(8, 12))


barWidth = 0.25
r = np.arange(len(wordcloud))
ax[0].bar(range(len(lummy_counts)), lummy_counts.values(), color='skyblue', label="Lummy Counts")
ax[0].set_title("Occurrences of 'lummy' per Participant")
ax[0].set_ylabel("Count")
ax[0].set_xticks(range(len(wordcloud)))
ax[0].set_xticklabels(wordcloud.keys())
ax[0].grid(axis='y')

ax[1].bar(range(len(wordcloud)), word_counts.values(), color='red', label="Word Counts")
ax[1].set_title("Total wordcounts")
ax[1].set_xticks(range(len(wordcloud)))
ax[1].set_xticklabels(wordcloud.keys())
ax[1].grid(axis='y')

ax[2].bar(range(len(wordcloud)), lummy_percentages.values(), color='green', label="Lummy Percentages")
ax[2].set_xticks(range(len(wordcloud)))
ax[2].set_xticklabels(wordcloud.keys() )
ax[2].set_ylabel("Proportion")
ax[2].grid(axis='y')

#ax.set_xticks(r + barWidth)
plt.savefig("images/lummy_bar_chart.png", bbox_inches='tight')


print("got to wordcloud")
# plot wordclouds
plt.figure(dpi=10000)
fig, axis = plt.subplots(2, round(len(wordcloud)/2)+1, figsize=(60,25))
for index,participant in enumerate(wordcloud):
    print(index%2," ",round(index/2.01))
    wordcloud_fig = WordCloud(width = 3840, height = 2160, max_words=200, background_color="white", collocations=True).generate((" ").join(wordcloud[participant]))
    axis[(index)%2,round(index/2.01)].imshow(wordcloud_fig)
    axis[(index)%2,round(index/2.01)].axis("off")
    axis[(index)%2,round(index/2.01)].set_title(participant, fontsize=30)
    #plt.imshow(wordcloud_fig)
    #plt.title(participant)
    #plt.axis("off")
    #plt.savefig("images/"+participant+".png", bbox_inches='tight')
axis[1,2].bar(range(len(wordcloud)), word_counts.values(), color='red', label="Word Counts")
axis[1,2].set_title("Total wordcounts", fontsize=50)
axis[1,2].set_xticks(range(len(wordcloud)))
axis[1,2].set_xticklabels(wordcloud.keys(),fontsize=25)
axis[1,2].grid(axis='y', zorder=0)

plt.savefig("images/purgers.png", bbox_inches='tight')
plt.show()
plt.close()      

with open("processed_messages.json", mode="w", encoding="utf-8") as write_file:
    json.dump(messages, write_file, ensure_ascii=False, indent=4)
    
    