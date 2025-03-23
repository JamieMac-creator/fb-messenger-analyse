WIN_USER="jamie"

DIR="/mnt/c/MyFiles/PersonalProjects/fb/2025dump"
PATTERN="facebook-*"

process_file() {
    local file="$1"
    echo "Processing $file"
    unzip -q -n ${file} *.json -d "data" 2>/dev/null
    # Add your processing logic here
}

if [ ! -d "$DIR" ]; then
    echo "Directory $DIR not found."
    exit 1
fi

# Get the facebook data, before e2ee cutover
for file in "$DIR"/$PATTERN; do
    if [ -f "$file" ]; then
        process_file "$file"
    fi
done

# Get the facebook data, after e2ee cutover
unzip -q -n "${DIR}/messages.zip" *.json -d "data/e2e"

rename "s/ /_/g" data/*/*