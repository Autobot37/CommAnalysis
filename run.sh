#!/bin/bash

# Defaults
DEFAULT_FOLDER="data/gsocvideos"
DEFAULT_ST="chunk"
DEFAULT_WSZ=1
DEFAULT_GDRIVE=""
DEFAULT_HF_TOKEN="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"
DEFAULT_CT=5

# Initialize variables
FOLDER="$DEFAULT_FOLDER"
ST="$DEFAULT_ST"
WSZ="$DEFAULT_WSZ"
GDRIVE="$DEFAULT_GDRIVE"
HF_TOKEN="$DEFAULT_HF_TOKEN"
CT="$DEFAULT_CT"

POSITIONAL_ARGS=()

# Parsing loop
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --folder)
            if [[ -n "$2" && "$2" != --* ]]; then FOLDER="$2"; shift 2;
            else echo "Error: Argument for $1 is missing" >&2; exit 1; fi
            ;;
        --st)
            if [[ -n "$2" && "$2" != --* ]]; then ST="$2"; shift 2;
            else echo "Error: Argument for $1 is missing" >&2; exit 1; fi
            ;;
        --wsz)
            if [[ -n "$2" && "$2" != --* ]]; then WSZ="$2"; shift 2;
            else echo "Error: Argument for $1 is missing" >&2; exit 1; fi
            ;;
        --gdrive)
            if [[ -n "$2" && "$2" != --* ]]; then GDRIVE="$2"; shift 2;
            else echo "Error: Argument for $1 is missing" >&2; exit 1; fi
            ;;
        --hf_token) # Added
            if [[ -n "$2" && "$2" != --* ]]; then HF_TOKEN="$2"; shift 2;
            else echo "Error: Argument for $1 is missing" >&2; exit 1; fi
            ;;
        --ct) # Added
            if [[ -n "$2" && "$2" != --* ]]; then CT="$2"; shift 2;
            else echo "Error: Argument for $1 is missing" >&2; exit 1; fi
            ;;
        -h|--help) # Updated help
            echo "Usage: $0 [options] [folder_path]"
            echo "Options:"
            echo "  --folder <path>    : Path to video folder (Default: $DEFAULT_FOLDER)"
            echo "  --st <type>        : Analysis Strategy: 'chunk' or 'sentence' (Default: $DEFAULT_ST)"
            echo "  --wsz <num>        : Window size for analysis (Default: $DEFAULT_WSZ)"
            echo "  --ct <num>         : Chunk time in seconds for 'chunk' strategy (Default: $DEFAULT_CT)"
            echo "  --gdrive <url>     : Google Drive folder URL (Optional)"
            echo "  --hf_token <token> : Hugging Face Hub token (Default: <provided>)"
            echo "  -h, --help         : Display this help message"
            exit 0
            ;;
        --)
            shift; break ;;
        -*)
            echo "Unknown option: $1" >&2; exit 1 ;;
        *)
            POSITIONAL_ARGS+=("$1"); shift ;;
    esac
done

POSITIONAL_ARGS+=("$@")

if [[ ${#POSITIONAL_ARGS[@]} -gt 0 && "$FOLDER" == "$DEFAULT_FOLDER" ]]; then
    FOLDER="${POSITIONAL_ARGS[0]}"
fi

if [[ ${#POSITIONAL_ARGS[@]} -gt 1 ]]; then
    echo "Warning: Multiple positional arguments provided. Using '$FOLDER' for folder." >&2
fi

# Set PYTHONPATH so Python can find the 'src' module
export PYTHONPATH=$(pwd)

# Build command array, including new arguments
CMD_ARGS=(python3 src/main.py "$FOLDER" --st "$ST" --wsz "$WSZ" --ct "$CT" --hf_token "$HF_TOKEN")

# Conditionally add --gdrive if it has a value
if [ -n "$GDRIVE" ]; then
  CMD_ARGS+=(--gdrive "$GDRIVE")
fi

echo "Executing:"
printf "%q " "${CMD_ARGS[@]}"
echo

"${CMD_ARGS[@]}"

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Python script failed with exit code $EXIT_CODE" >&2
    exit $EXIT_CODE
fi

echo "Script finished successfully."
exit 0