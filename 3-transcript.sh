#!/bin/zsh

# For now we will assume its a groupchat and that the users stay static
CHAT_DIR="data/inbox/asspurgers_1559279850760052"
# CHAT_DATA_PATTERN="test_message.json"

# First sed: as facebook uses incorrect emoji format we need to correct it
# Second sed: remove newlines
# Third sed: remove awkward backslash from messages
# Fourth sed: remove percent signs

for file in ${CHAT_DIR}/*.json; do
    echo "Processing $file"
    cat ${file} | sed -E 's#\\u00([0-9a-f]{2})#\\x\L\1#g' | \
        sed 's#\\[nrt]# #g' | \
        sed 's#\\.##g'| \
        sed 's#\\.##g'| \
        > sedded.json
    sedded=$(cat sedded.json)
    if [ $(basename ${file}) = "message_1.json" ]; then
        echo "$sedded" > messages_to_parse.json
    else
        echo "$sedded" > jq .
        # TODO: get rid of jq
        echo "$sedded" | jq -s '.[0] * {messages: (.[0].messages + .[1].messages)}' messages_to_parse.json - > temp_messages_to_parse.json
        mv temp_messages_to_parse.json messages_to_parse.json
    fi
done

#rm sedded.json

# Now pass off to a python script to complete analysis



# We are interested in two categories of messages:
# 1. Messages with content - "content" field is not null
# 2. Messages with images - "photos" field is not null
# 3. Unsent messages - "is_unsent" field is true
# 4. Stickers, gifs, audio_files, etc
# 5. Reactions - count 5 reaction messages, 4 reaction messages, and any with a self-reaction. Count total given and recieved
#
# Note these may have overlap, in which case we should delete the irrelevant field in each
# If a message has neither we don't care about it