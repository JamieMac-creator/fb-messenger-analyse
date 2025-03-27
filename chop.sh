#!/bin/bash
IMAGE=

# Check if the input file exists
if [ ! -f "image.jpg" ]; then
    echo "Error: image.jpg not found!"
    exit 1
fi

# Get the image dimensions
height=$(identify -format "%h" image.jpg)
width=$(identify -format "%w" image.jpg)

# Calculate the height of each strip
strip_height=$((height / 5))

# Chop the image into 5 horizontal strips
for i in {0..4}; do
    offset=$((i * strip_height))
    convert image.jpg -crop "${width}x${strip_height}+0+${offset}" "strip_$((i + 1)).png"
done

echo "Image chopped into 5 horizontal strips: strip_1.png to strip_5.png"