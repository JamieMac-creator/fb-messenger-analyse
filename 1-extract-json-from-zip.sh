DIR="/mnt/c/MyFiles/PersonalProjects/fb/2025dump"
PATTERN="facebook-*"

if [ ! -d "$DIR" ]; then
    echo "Directory $DIR not found."
    exit 1
fi

# Get non-e2ee messages
for file in "$DIR"/$PATTERN; do
    if [ -f "$file" ]; then
        echo "Processing $file"
        unzip -q -n ${file} *.json -d "data" 2>/dev/null
    fi
done

# Get e2ee messages
unzip -q -n "${DIR}/messages.zip" *.json -d "data/e2e"

# Remove spaces
rename "s/ /_/g" data/*/*

cp -r data/your_facebook_activity/messages/inbox data/inbox
cp -r data/your_facebook_activity/messages/e2ee_cutover data/e2ee_cutover
rm -rf data/your_facebook_activity