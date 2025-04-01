#!/bin/zsh

DATA_DIR="data/"

mkdir -p cleaned

for file in ${DATA_DIR}/**/*.json; do
    echo "Processing ${file}"
    # First sed: as facebook uses incorrect emoji format we need to correct it
    # Second sed: remove newlines
    # Third sed: remove awkward backslash from messages
    cat ${file} | sed -E 's#\\u00([0-9a-f]{2})#\\x\L\1#g' | \
        sed 's#\\[nrt\\]# #g' | \
        sed 's#\\",$#",#g' \
        > sedded.json
    sedded=$(cat sedded.json)
    echo "${sedded}" > ${file}
done

rm sedded.json