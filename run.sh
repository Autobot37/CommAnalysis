#!/bin/bash

# Default values
FOLDER="data/gsocvideos"
ST="chunk"
WSZ=1

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --folder)
            FOLDER="$2"
            shift
            ;;
        --st)
            ST="$2"
            shift
            ;;
        --wsz)
            WSZ="$2"
            shift
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift
done

# Run the Python script with the parsed or default arguments.
PYTHONPATH=$(pwd) python3 src/main.py "$FOLDER" --st "$ST" --wsz "$WSZ"
