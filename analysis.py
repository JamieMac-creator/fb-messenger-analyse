import json
from datetime import datetime

#Set the backend to use mplcairo
import matplotlib, mplcairo
print('Default backend: ' + matplotlib.get_backend())
matplotlib.use("module://mplcairo.qt")
print('Backend is now ' + matplotlib.get_backend())

import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
fpath = Path("./NotoColorEmoji-Regular.ttf")
from wordcloud import WordCloud
import numpy as np
from collections import defaultdict

chat_name = "asspurgers_1559279850760052"
with open(str("cleaned/data/inbox/"+chat_name+"/clean_"+chat_name+".json"), mode="r", encoding="utf-8") as read_file:
    messages_json = json.load(read_file, strict=False)

# Make list of participants
participants = []
for participant in messages_json["participants"]:
    participants.append(participant["name"])
print(participants)

def remove_heathens():
    # Remove people who have left the group
    filtered_messages = []
    for message in messages_json["messages"]:
        if message["timestamp_ms"] == 1514196622537:
            print("This is a test message")
            print(json.dumps(message, indent=2))

        if message["sender_name"] not in participants:
            continue
        else:
            if message["timestamp_ms"] == 1514196622537:
                print("else block hit")
            # Remove reactions from this user
            if "reactions" in message:
                valid_reactions = [reaction for reaction in message["reactions"] if reaction["actor"] in participants]
                if message["timestamp_ms"] == 1514196622537:
                    print("This is the valid_reactions:")
                    print(json.dumps(valid_reactions, indent=2))
                message["reactions"] = valid_reactions
        filtered_messages.append(message)
        if message["timestamp_ms"] == 1514196622537:
            print("2222This is a test message")
            print(json.dumps(message, indent=2))
    messages_json["messages"] = filtered_messages
remove_heathens()

def timestamp_fixing():
    for message in messages_json["messages"]:
        message['datetime'] = datetime.fromtimestamp(message['timestamp_ms'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
    messages_json["messages"].sort(key=lambda x: x['timestamp_ms'])
    return 0

timestamp_fixing()

# Collect messages reacted to by everyone
def collect_fully_reacted():
    fully_reacted_messages = []
    for message in messages_json["messages"]:
        if "reactions" in message:
            for participant in participants:
                if (participant not in [reaction["actor"] for reaction in message["reactions"]]) and (participant != message["sender_name"]):
                    break
            else:
                fully_reacted_messages.append(message)
    return fully_reacted_messages

def collect_messages_once_per_reaction():
    duplicate_reacted_messages = []
    for message in messages_json["messages"]:
        if "reactions" in message:
            for participant in participants:
                if (participant in [reaction["actor"] for reaction in message["reactions"]]):
                    duplicate_reacted_messages.append(message)
    return duplicate_reacted_messages

def count_recieved_reactions():
    # Create a dictionary to count reactions received by each participant
    reactions_received = defaultdict(int)

    # Iterate through the messages
    for message in messages_json["messages"]:
        if "reactions" in message:
            # Count the reactions for each participant
            for reaction in message["reactions"]:
                if reaction["actor"] != message["sender_name"]:
                    reactions_received[message["sender_name"]] += 1

    # Output as bar chart
    fig, ax = plt.subplots(dpi=300, figsize=(10, 8))
    ax.bar(reactions_received.keys(), reactions_received.values(), color='skyblue')
    ax.set_title("Reactions Received by Each Participant")
    ax.set_ylabel("Count")
    ax.set_xticks(range(len(reactions_received)))
    ax.set_xticklabels(reactions_received.keys(), rotation=45, ha="right")
    ax.grid(axis='y')
    plt.savefig("images/reactions_received.png", bbox_inches='tight')
    plt.close()

def annual_frequency_per_hour_stacked_bar_chart(messages_param, outfigname, figtitle, normalize=True):
    # Group messages by hour and year
    message_counts_by_year = defaultdict(lambda: defaultdict(int))
    for message in messages_param:
        hour = datetime.fromtimestamp(message["timestamp_ms"] / 1000).hour
        year = datetime.fromtimestamp(message["timestamp_ms"] / 1000).year
        message_counts_by_year[hour][year] += 1

    # Prepare data for stacked bar chart
    hours = np.arange(24)
    years = sorted({year for hour in message_counts_by_year for year in message_counts_by_year[hour]})
    data = {year: [message_counts_by_year[hour].get(year, 0) for hour in hours] for year in years}

    # Normalize data if required
    if normalize:
        total_counts = np.sum([np.array(data[year]) for year in years], axis=0)
        for year in years:
            data[year] = [count / total if total > 0 else 0 for count, total in zip(data[year], total_counts)]

    # Create a stacked bar chart with gradient colors
    fig, ax = plt.subplots(dpi=300, figsize=(10, 8))
    cmap = plt.get_cmap('RdYlGn', len(years))  # Gradient color map

    bottom = np.zeros(len(hours))
    for i, year in enumerate(years):
        ax.bar(hours, data[year], bottom=bottom, label=str(year), color=cmap(i), zorder=3)
        bottom += np.array(data[year])

    ax.set_title(figtitle)
    ax.set_ylabel("Count" if not normalize else "Proportion")
    ax.set_xticks(range(len(hours)))
    ax.set_xticklabels(hours, rotation=45, ha="right")
    ax.legend(reverse=True, title="Year")
    ax.grid(axis='y', zorder=0)

    # Save the chart
    plt.savefig("images/"+outfigname+"_messages_stacked_bar_chart.png", bbox_inches='tight')
    plt.close()

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
    cmap = plt.get_cmap('RdYlGn', len(years))  # Gradient color map

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
    plt.close()

def reactions_given_by_users(messages_param, outfigname, figtitle, top_n=5, stackby="content"):
    # Group messages by sender and content
    content_count_dict = defaultdict(lambda: defaultdict(int))
    for message in messages_param:
        sender = message["sender_name"]
        content = message.get(stackby, "No Content")
        content_count_dict[sender][content] += 1

    # Although the dict of dicts is required pre-sorting, afterwards what we actually need is a list of dicts of dicts
    # The outer dict is the sender, the inner dict is the content and count
    content_data = []
    for i in range(top_n+1):
        content_data.append({})
    
    # For each sender, identify the top N contents
    for sender in content_count_dict:
        # Sort by count. We have to place it in an intermediate var as sorted needs to work on tuples (to be readable AFAIK)
        sorted_tuples = sorted(content_count_dict[sender].items(), key=lambda x: x[1], reverse=True)
        # However we convert in into a list of dicts
        content_sorted_dicts = [{"content": content, "count": count} for content, count in sorted_tuples]
        # Not all people will have 10 different reactions, so we need to pad the list with empty dicts
        for i in range(len(content_sorted_dicts), top_n+2):
            content_sorted_dicts.append({"content": "No Content", "count": 0})
        # For every reaction outside top 10, transfer to "Other"
        content_sorted_dicts[top_n]["content"] = "Other"
        for unpopular_dict in content_sorted_dicts[top_n+1:]:
            content_sorted_dicts[top_n]["count"] += unpopular_dict["count"]
        content_sorted_dicts = content_sorted_dicts[:top_n+1]
        #print(content_sorted_dicts)

        # Now we need to reshape this data to be used in a bar chart
        for i in range(top_n+1):
            content_data[i][sender] = content_sorted_dicts[i]

    # Prepare data for stacked bar chart
    senders = list(content_count_dict.keys())
    
    # Create a stacked bar chart with gradient colors
    fig, ax = plt.subplots(dpi=300, figsize=(10, 8))
    cmap = plt.get_cmap("viridis", top_n+1)  # Gradient color map
    bottom = np.zeros(len(senders))
    
    # Create the stacked bar chart layer by layer
    for indx,layer in enumerate(content_data):
        # Need to create a map of favourite emojis, then 2nd favourite etc
        counts = [layer[sender]["count"] for sender in senders]
        contents = [layer[sender]["content"] for sender in senders]
        p = ax.bar(senders, counts, bottom=bottom, color=cmap(indx), zorder=3)
        if indx < top_n:
            if stackby == "content":
                ax.bar_label(p, labels=contents, label_type='center', font=fpath)
            else:
                ax.bar_label(p, labels=counts, label_type='center')
        bottom += np.array(counts)
        # Set last bar label to be normal font
    ax.bar_label(p, labels=contents, label_type='center', padding=10)
    ax.set_title(figtitle)
    ax.set_ylabel("Count")
    ax.set_xticks(range(len(senders)))
    ax.set_xticklabels(senders, rotation=45, ha="right")
    ax.grid(axis='y', zorder=0)

    # Save the chart
    plt.savefig("images/"+outfigname+"_content_stacked_bar_chart.png", bbox_inches='tight')
    plt.close()

def turn_reactions_into_messages():
    reaction_messages = []
    for message in messages_json["messages"]:
        if "reactions" in message:
            for reaction in message["reactions"]:
                reaction_message = {
                    "sender_name": reaction["actor"],
                    "timestamp_ms": message["timestamp_ms"],
                    "content": reaction["reaction"],
                    "reaction_type": reaction["reaction"],
                    "reaction_target": message["sender_name"]
                }
                reaction_messages.append(reaction_message)
    return reaction_messages

def work_out_relative_reaction_dist():
    # array is built with sender (of reaction) outside and recipient (sender of msg) inside
    reactions_real = np.zeros((len(participants), len(participants)), dtype=int)
    participant_lookup = {participant: i for i, participant in enumerate(participants)}
    # Populate real reaction array
    for message in messages_json["messages"]:
        if "reactions" in message:
            for reaction in message["reactions"]:
                reactions_real[participant_lookup[message["sender_name"]]][participant_lookup[reaction["actor"]]] += 1
    # Remove self-reacts
    np.fill_diagonal(reactions_real, 0)
    print("reactions_real:\n" + str(reactions_real))
    # Create "expected" reaction array assuming independent distribution
    # NOTE axis 0 is the row you are in, axis 1 is the column you are in
    # imagine this as a sum along the bottom
    reactions_given = np.sum(reactions_real, axis=0, keepdims=True)
    # imagine this as a column along the side
    reactions_received = np.sum(reactions_real, axis=1, keepdims=True)
    reactions_total = np.sum(reactions_real)
    
    # Proceed only if all of reactions_given and reactions_received are nonzero
    if np.any(reactions_given == 0) or np.any(reactions_received == 0):
        print("Some participants have zero reactions given or received. Skipping further calculations.")
        return
    # Note a simple outer won't be accurate as it assumes people react to themselves
    # with equal frequncy as to others, which is not true
    # To fix this we "redistribute" expected self-reacts using an algorithm
    reactions_expected = np.outer(reactions_received, reactions_given) / reactions_total
    np.fill_diagonal(reactions_expected, 0)
    # Apply the Iterative Proportional Fitting (IPF) method
    for i in range(10): 
        # Adjust rows
        row_sums_current_as_col = reactions_expected.sum(axis=1, keepdims=True)
        # row scaling is the proportion of the row sum of each exp value so far
        row_scaling = reactions_received / row_sums_current_as_col
        # print("row_scaling " + str(i) + ": " + str(row_scaling))
        reactions_expected *= row_scaling

        # Adjust cols
        col_sums = reactions_expected.sum(axis=0, keepdims=True)
        col_scaling = reactions_given / col_sums
        # print("col_scaling "+ str(i) +": " + str(col_scaling))
        reactions_expected *= col_scaling
    print("reactions_expected:\n" + str(reactions_expected))
    np.fill_diagonal(reactions_expected, 1)
    reactions_diff = (reactions_real - reactions_expected).astype(int)
    reactions_percents_diff = ((reactions_real/reactions_expected) - 1) * 100
    np.fill_diagonal(reactions_percents_diff, 0)
    np.fill_diagonal(reactions_expected, 0)
    # print("Users are: " + str(participant_lookup))
    # print("Total reactions: " + str(reactions_given.sum()))
    # print("Reactions given by users: " + str(reactions_given))
    # print("Reactions recieved by users: " + str(reactions_received))
    # reactions_real[1,3] = 5000
    # print("Reactions REAL\n" + str(reactions_real))
    # print("Reactions proportion recieved by users:\n " + str(reactions_received / reactions_given.sum()))
    # print("Reactions expected by users:")
    # print(reactions_expected)
    # print("Reactions expected by users (column sum):")
    # print(reactions_expected.sum(axis=0))
    # print("Reactions expected by users (row sum):")
    # print(reactions_expected.sum(axis=1))
    # print("Reactions diff:")
    # print(reactions_percents_diff)
    # print(reactions_percents_diff.sum())
    # Create a heatmap to visualize the reactions_percents_diff array
    fig, ax = plt.subplots(dpi=300, figsize=(10, 8))
    cmap = mpl.colors.ListedColormap(['red', 'white', 'green'])
    bounds = [np.min(reactions_percents_diff), -1, 1, np.max(reactions_percents_diff)]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # Plot the heatmap
    im = ax.imshow(reactions_percents_diff, cmap='RdYlGn', vmin=-100, vmax=100)

    # Add participant names as axis labels
    ax.set_xticks(range(len(participants)))
    ax.set_yticks(range(len(participants)))
    ax.set_xticklabels(participants, rotation=45, ha="right")
    ax.set_yticklabels(participants)
    ax.set_xlabel("Reactions Given")
    ax.set_ylabel("Reactions Received")

    # Annotate the heatmap with values
    for i in range(len(participants)):
        for j in range(len(participants)):
            text_color = "green" if reactions_percents_diff[i, j] > 0 else "red" if reactions_percents_diff[i, j] < 0 else "black"
            ax.text(j, i, f"{'+' if reactions_percents_diff[i, j] > 0 else ''}{reactions_percents_diff[i, j]:.1f}%", ha="center", va="center", color="black", fontsize=80/len(participants))
            ax.text(j, i + 0.2, f"{'+' if reactions_diff[i, j] > 0 else ''}{reactions_diff[i, j]}", ha="center", va="center", color="grey", fontsize=40/len(participants))
            # Set diagonal of the grid to black fill
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, color='grey', zorder=2))

    # Set title and save the heatmap
    ax.set_title("Reaction Difference (compared to independent assumption) Heatmap")
    plt.savefig("images/reactions_diff_heatmap.png", bbox_inches='tight')
    plt.close()
    
work_out_relative_reaction_dist()
########################################################################################################################
# This bit actually outputs the graphs

count_recieved_reactions()

# Analyze messages by hour of the day
def messages_by_hour():
    # Create a dictionary to count messages per hour
    hour_counts = defaultdict(int)

    # Iterate through the messages
    for message in messages_json["messages"]:
        hour = datetime.fromtimestamp(message["timestamp_ms"] / 1000).hour
        hour_counts[hour] += 1

    # Prepare data for plotting
    hours = list(range(24))
    counts = [hour_counts.get(hour, 0) for hour in hours]

    # Plot the data
    fig, ax = plt.subplots(dpi=300, figsize=(10, 6))
    ax.bar(hours, counts, color='skyblue')
    ax.set_title("Messages by Hour of the Day")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Message Count")
    ax.set_xticks(hours)
    ax.grid(axis='y')

    # Save the chart
    plt.savefig("images/messages_by_hour.png", bbox_inches='tight')
    plt.close()

# Call the function
messages_by_hour()

# Create some helper arrays related to reactions
fully_reacted_messages = collect_fully_reacted()
print(fully_reacted_messages)
reaction_multiple_messages = collect_messages_once_per_reaction()
reactions_as_messages = turn_reactions_into_messages()

reactions_given_by_users(reactions_as_messages, "reactions_sorted_by_occur", "Reactions given", top_n=8)
reactions_given_by_users(reactions_as_messages, "reactions_by_target", "Reactions given split by target", top_n=4, stackby="reaction_target")

# Generate graphs for a bunch of different things
annual_frequency_per_user_stacked_bar_chart(fully_reacted_messages, "fully_reacted", "Fully Reacted Messages")
annual_frequency_per_user_stacked_bar_chart(messages_json["messages"], "messages_sent", "Total Messages Sent")
annual_frequency_per_hour_stacked_bar_chart(messages_json["messages"], "messages_by_hour", "Total Messages Sent By Hour", normalize=False)
annual_frequency_per_hour_stacked_bar_chart(messages_json["messages"], "messages_normalised_by_hour", "Total Messages Sent By Hour", normalize=True)
reaction_arr = turn_reactions_into_messages()
annual_frequency_per_user_stacked_bar_chart(reaction_arr, "reactions", "Reactions given")
# Only select nonzero content messages
nonzero_content_messages = [message for message in messages_json["messages"] if "content" in message and message["content"]]
annual_frequency_per_user_stacked_bar_chart(nonzero_content_messages, "nonzero_content", "Nonzero Content Messages")

# Only select images
image_messages = [message for message in messages_json["messages"] if "photos" in message]
annual_frequency_per_user_stacked_bar_chart(image_messages, "images", "Image Messages")

# Only select is_unsent messages
unsent_messages = [message for message in messages_json["messages"] if "is_unsent" in message]
annual_frequency_per_user_stacked_bar_chart(unsent_messages, "unsent", "Unsent Messages")

with open("processed_messages.json", mode="w", encoding="utf-8") as write_file:
    json.dump(messages_json, write_file, ensure_ascii=False, indent=4)

quit()

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
