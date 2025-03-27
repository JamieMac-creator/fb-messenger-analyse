import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from wordcloud import WordCloud
import numpy as np
from collections import defaultdict

with open("cleaned_messages.json", mode="r", encoding="utf-8") as read_file:
    messages_json = json.load(read_file, strict=False)

# Make list of participants
participants = []
for participant in messages_json["participants"]:
    participants.append(participant["name"])

def timestamp_fixing():
    for message in messages_json["messages"]:
        message['datetime'] = datetime.fromtimestamp(message['timestamp_ms'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
    messages_json["messages"].sort(key=lambda x: x['timestamp_ms'])
    return 0
timestamp_fixing()

fully_reacted_messages = []

# Collect messages reacted to by everyone
def collect_fully_reacted():
    for message in messages_json["messages"]:
        if "reactions" in message:
            for participant in participants:
                if (participant not in [reaction["actor"] for reaction in message["reactions"]]) and (participant != message["sender_name"]):
                    break
            #else:  # Executes only if the loop wasn't broken
                print(f"{message['sender_name']}: {message.get('content', 'No content')}")
                fully_reacted_messages.append(message)
    return fully_reacted_messages

def collect_messages_once_per_reaction():
    for message in messages_json["messages"]:
        if "reactions" in message:
            for participant in participants:
                if (participant in [reaction["actor"] for reaction in message["reactions"]]):
                    fully_reacted_messages.append(message)
    return fully_reacted_messages

fully_reacted_messages = collect_fully_reacted()
reaction_multiple_messages = collect_messages_once_per_reaction()

def annual_frequency_per_user_stacked_bar_chart(messages_param, outfigname, figtitle):
    # Group messages by sender and year
    message_counts_by_year = defaultdict(lambda: defaultdict(int))
    for message in messages_param:
        sender = message["sender_name"]
        year = datetime.fromtimestamp(message["timestamp_ms"] / 1000).year
        message_counts_by_year[sender][year] += 1

    # Prepare data for stacked bar chart
    senders = list(message_counts_by_year.keys())
    years = sorted({year for sender in message_counts_by_year for year in message_counts_by_year[sender]})
    data = {year: [message_counts_by_year[sender].get(year, 0) for sender in senders] for year in years}

    # Create a stacked bar chart with gradient colors
    fig, ax = plt.subplots(dpi=300, figsize=(10, 8))
    cmap = cm.get_cmap('RdYlGn', len(years))  # Gradient color map

    bottom = np.zeros(len(senders))
    for i, year in enumerate(years):
        ax.bar(senders, data[year], bottom=bottom, label=str(year), color=cmap(i), zorder=3)
        bottom += np.array(data[year])

    ax.set_title(figtitle)
    ax.set_ylabel("Count")
    ax.set_xticks(range(len(senders)))
    ax.set_xticklabels(senders, rotation=45, ha="right")
    ax.legend(reverse=True, title="Year")
    ax.grid(axis='y', zorder=0)

    # Save the chart
    plt.savefig("images/"+outfigname+"_messages_stacked_bar_chart.png", bbox_inches='tight')
    plt.show()

def annual_frequency_top_user_stacked_bar_chart(messages_param, outfigname, figtitle):
    import matplotlib.cm as cm

    # Group messages by sender and year
    message_counts_by_year = defaultdict(lambda: defaultdict(int))
    for message in messages_param:
        sender = message["sender_name"]
        year = datetime.fromtimestamp(message["timestamp_ms"] / 1000).year
        message_counts_by_year[sender][year] += 1

    # Prepare data for stacked bar chart
    senders = list(message_counts_by_year.keys())
    years = sorted({year for sender in message_counts_by_year for year in message_counts_by_year[sender]})
    data = {year: [message_counts_by_year[sender].get(year, 0) for sender in senders] for year in years}

    # Calculate total messages per sender
    total_messages_per_sender = {sender: sum(message_counts_by_year[sender].values()) for sender in senders}
    highest_sender = max(total_messages_per_sender, key=total_messages_per_sender.get)
    average_sender_data = {year: sum(data[year]) / (len(senders) - 1) for year in years}

    # Create a grouped stacked bar chart for the highest sender and average user
    fig, ax = plt.subplots(dpi=300, figsize=(10, 8))
    cmap = cm.get_cmap('viridis', len(years))  # Gradient color map

    x = np.arange(len(years))  # X-axis positions for the groups
    bar_width = 0.4  # Width of each bar

    # Prepare data for the highest sender
    highest_sender_data = [message_counts_by_year[highest_sender].get(year, 0) for year in years]

    # Prepare data for the average user (excluding the top sender)
    average_sender_data = [
        sum(data[year]) / (len(senders) - 1) if len(senders) > 1 else 0 for year in years
    ]

    # Create a stacked bar chart
    ax.bar(
        [0], [sum(highest_sender_data)], label=f"{highest_sender}", color="blue"
    )
    ax.bar(
        [1], [sum(average_sender_data)], label="Average User", color="orange"
    )

    ax.set_title(figtitle)
    ax.set_ylabel("Count")
    ax.set_xlabel("Year")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.legend(title="User")
    ax.grid(axis='y')

    # Save the chart
    plt.savefig("images/"+outfigname+"_messages_stacked_bar_chart.png", bbox_inches='tight')
    plt.show()

# Example call
annual_frequency_per_user_stacked_bar_chart(fully_reacted_messages, "fully_reacted", "Fully Reacted Messages")
annual_frequency_per_user_stacked_bar_chart(messages_json["messages"], "messages_sent", "Total Messages Sent")
annual_frequency_top_user_stacked_bar_chart(reaction_multiple_messages, "reaction_multiple", "Total reactions received")

# Only select nonzero content messages
nonzero_content_messages = [message for message in messages_json["messages"] if "content" in message and message["content"]]
annual_frequency_per_user_stacked_bar_chart(nonzero_content_messages, "nonzero_content", "Nonzero Content Messages")

# Only 
def print():
    print(Hello World)
    


# Only select contentless messages
zero_content_messages = [
    message for message in messages_json["messages"] 
    if "content" not in message and "photos" not in message and "is_unsent" not in message and "sticker" not in message and "videos" not in message and "audio_files" not in message and "gifs" not in message and "files" not in message
]
annual_frequency_per_user_stacked_bar_chart(zero_content_messages, "zero_content", "Zero Content Messages")

# Only select images
image_messages = [message for message in messages_json["messages"] if "photos" in message]
annual_frequency_per_user_stacked_bar_chart(image_messages, "images", "Image Messages")

# Only select is_unsent messages
unsent_messages = [message for message in messages_json["messages"] if "is_unsent" in message]
annual_frequency_per_user_stacked_bar_chart(unsent_messages, "unsent", "Unsent Messages")

with open("contentless_messages.json", mode="w", encoding="utf-8") as write_file:
    json.dump(zero_content_messages, write_file, ensure_ascii=False, indent=4)

with open("processed_messages.json", mode="w", encoding="utf-8") as write_file:
    json.dump(messages_json, write_file, ensure_ascii=False, indent=4)

############################################################################################################

# Create a single wordcloud for all fully reacted messages
def generate_fully_reacted_wordcloud():
    all_words = []

    # Collect words from fully reacted messages
    for message in fully_reacted_messages:
        if 'content' in message:
            all_words.extend(message['content'].split())

    # Generate and plot the wordcloud
    plt.figure(dpi=300)
    wordcloud_fig = WordCloud(width=1600, height=800, max_words=200, background_color="white").generate(" ".join(all_words))
    plt.imshow(wordcloud_fig, interpolation="bilinear")
    plt.axis("off")
    plt.title("Wordcloud for Fully Reacted Messages", fontsize=16)
    plt.savefig("images/fully_reacted_wordcloud.png", bbox_inches='tight')
    plt.show()

# Call the function
generate_fully_reacted_wordcloud()


quit()
wordcloud = {}
for participant in messages["participants"]:
    wordcloud[participant["name"]] = []
# For each message dict in list of messages
for message in messages["messages"]:
    # Create human readable timestamp
    if 'timestamp_ms' in message:
        message['datetime'] = datetime.fromtimestamp(message['timestamp_ms'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
    # Delete bogus keys
    if 'is_geoblocked_for_viewer' in message:
        del message['is_geoblocked_for_viewer']
    if 'is_unsent_image_by_messenger_kid_parent' in message:
        del message['is_unsent_image_by_messenger_kid_parent']
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


    
    
plt.close()      


    
    
