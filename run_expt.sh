#!/bin/bash
#SBATCH --job-name=run_expt
#SBATCH --output=run_expt.out
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=3-0

# Shell script to run main.py for all topic file names and models
# Usage: ./run_all_topics.sh

# using batch API, set to true or false
USE_BATCH_API=false

# List of all topic file names
TOPICS=(
    "Age"
    "Disability_status"
    "Gender_identity"
    "Nationality"
    "Physical_appearance"
    "Race_ethnicity"
    "Race_x_gender"
    "Race_x_SES"
    "Religion"
    "SES"
    "Sexual_orientation"
)

# List of models to evaluate
MODELS=(
    # "gpt-4o-mini"
    # "gpt-3.5-turbo"
    # "gpt-4o"
    # "meta-llama/Llama-3-8b-chat-hf"
    # "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    # "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    # "google/gemma-3n-E4B-it"
    "gemini-2.5-flash"
)

# Output root directory
OUTPUT_ROOT_DIR="experiment_results"

# Model name mapping dictionary
declare -A MODEL_NAME_MAP
MODEL_NAME_MAP["gpt-4o-mini"]="gpt_4o_mini"
MODEL_NAME_MAP["gpt-3.5-turbo"]="gpt_3_5_turbo"
MODEL_NAME_MAP["gpt-4o"]="gpt_4o"
MODEL_NAME_MAP["meta-llama/Llama-3-8b-chat-hf"]="llama3_8b"
MODEL_NAME_MAP["google/gemma-3n-E4B-it"]="gemma_3n_e4b"
MODEL_NAME_MAP["meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"]="llama4_maverick_17b_128e"
MODEL_NAME_MAP["meta-llama/Llama-3.3-70B-Instruct-Turbo"]="llama3_3_70b"
MODEL_NAME_MAP["gemini-2.5-pro"]="gemini_2_5_pro"
MODEL_NAME_MAP["gemini-2.5-flash"]="gemini_2_5_flash"
MODEL_NAME_MAP["gemini-2.5-flash-lite"]="gemini_2_5_flash_lite"

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "Error: main.py not found in current directory"
    exit 1
fi

# Check if batch API is requested with non-OpenAI models
if [ "$USE_BATCH_API" = true ]; then
    for model in "${MODELS[@]}"; do
        if [[ ! "$model" == gpt* ]]; then
            echo "Warning: Batch API only works with OpenAI models (gpt-*)"
            echo "Model '$model' will cause errors with batch API"
            echo "Either set USE_BATCH_API=false or use only OpenAI models"
            exit 1
        fi
    done
fi

echo "Starting batch run at $(date)"
echo "Output root directory: $OUTPUT_ROOT_DIR"
echo "Topics: ${#TOPICS[@]}, Models: ${#MODELS[@]}"
if [ "$USE_BATCH_API" = true ]; then
    echo "Processing mode: Batch API (parallel processing)"
else
    echo "Processing mode: Sequential (one request at a time)"
fi
echo "================================"

# Counter for tracking progress
total_combinations=$((${#TOPICS[@]} * ${#MODELS[@]}))
current=0

# Run main.py for each topic and model combination
for topic in "${TOPICS[@]}"; do
    for model in "${MODELS[@]}"; do
        current=$((current + 1))
        
        # Get cleaned up model name from mapping
        MODEL_NAME_SHORT="${MODEL_NAME_MAP[$model]}"
        if [ -z "$MODEL_NAME_SHORT" ]; then
            echo "Warning: No mapping found for model '$model', using default cleanup"
            MODEL_NAME_SHORT=$(echo "$model" | sed 's/[^a-zA-Z0-9]/_/g')
        fi
        
        # Create output directory structure: OUTPUT_ROOT_DIR/MODEL_NAME_SHORT/topic
        OUTPUT_DIR="$OUTPUT_ROOT_DIR/$MODEL_NAME_SHORT"
        TOPIC_OUTPUT_DIR="$OUTPUT_DIR/$topic"
        mkdir -p "$OUTPUT_DIR"
        mkdir -p "$TOPIC_OUTPUT_DIR"
        
        echo "[$current/$total_combinations] Processing topic: $topic with model: $model"
        echo "Model: $MODEL_NAME_SHORT"
        echo "Output directory: $OUTPUT_DIR"
        echo "Topic directory: $TOPIC_OUTPUT_DIR"
        echo "Starting at $(date)"
        
        # MODIFIED SECTION: Conditional batch processing
        if [ "$USE_BATCH_API" = true ]; then
            echo "Using batch API (parallel processing)"
            echo "Note: Batch processing may take several minutes per example"
            # Run with batch API
            if python main.py "$topic" "$model" "$OUTPUT_DIR" --batch; then
                echo "✓ Successfully completed $topic with $model at $(date)"
            else
                echo "✗ Failed to process $topic with $model at $(date)"
            fi
        else
            echo "Using sequential processing (one request at a time)"
            # Run with sequential processing
            if python main.py "$topic" "$model" "$OUTPUT_DIR"; then
                echo "✓ Successfully completed $topic with $model at $(date)"
            else
                echo "✗ Failed to process $topic with $model at $(date)"
            fi
        fi
        # # Run the main.py with the current topic, model, and output directory
        # if python main.py "$topic" "$model" "$OUTPUT_DIR"; then
        #     echo "✓ Successfully completed $topic with $model at $(date)"
        # else
        #     echo "✗ Failed to process $topic with $model at $(date)"
        # fi
        
        echo "--------------------------------"
    done
done

echo "Batch run completed at $(date)"
echo "Total combinations processed: $current/$total_combinations"
