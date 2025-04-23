#!/bin/bash

# Colors for better readability
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Symbolic Regression test suite for datasets_bc...${NC}"

# Base parameters for all tests
BASE_PARAMS="--epochs 100 --start_population_size 1000 --population_size 100 --runs 5 --save_results"
FUNCTIONS="--use_functions sum_,neg_,mult_,div_,sin_,cos_,atan_,pow2_,pow3_"

# Create results directory if it doesn't exist
RESULTS_DIR="results"
if [ ! -d "$RESULTS_DIR" ]; then
    mkdir -p "$RESULTS_DIR"
    echo -e "${GREEN}Created results directory: $RESULTS_DIR${NC}"
fi

# Function to run tests with a header
run_test() {
    local test_name=$1
    local params=$2
    local log_file=$3
    
    echo -e "\n${YELLOW}====================================${NC}"
    echo -e "${YELLOW}Running test: ${test_name}${NC}"
    echo -e "${YELLOW}Parameters: ${params}${NC}"
    echo -e "${YELLOW}Log file: ${log_file}${NC}"
    echo -e "${YELLOW}====================================${NC}"
    
    # Execute the command with output redirection to log file
    python sr/sr_test.py ${params} > "${log_file}" 2>&1
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Test completed successfully!${NC}"
    else
        echo -e "${RED}Test failed!${NC}"
    fi
}

# Get all datasets from the datasets_bc folder
DATASETS=($(find datasets_bc -name "*.csv" -type f | sed 's|datasets_bc/||g' | sed 's|\.csv$||g'))

# Define error types and their combinations
declare -a ERROR_CONFIGS=(
    "Forward Error:--requires_forward_error"
    "Inverse Error:--requires_inv_error"
    "Absolute Inverse Error:--requires_abs_inv_error"
    "Spatial Absolute Inverse Error:--requires_spatial_abs_inv_error"
    "Forward + Inverse Error:--requires_forward_error --requires_inv_error"
    "Forward + Absolute Inverse Error:--requires_forward_error --requires_abs_inv_error"
    "Forward + Spatial Absolute Inverse Error:--requires_forward_error --requires_spatial_abs_inv_error"
    "Forward + Absolute + Spatial Absolute Error:--requires_forward_error --requires_abs_inv_error --requires_spatial_abs_inv_error"
    "All Error Types:--requires_forward_error --requires_inv_error --requires_abs_inv_error --requires_spatial_abs_inv_error"
)

# Counter for total tests and progress
TOTAL_TESTS=$((${#DATASETS[@]} * ${#ERROR_CONFIGS[@]}))
CURRENT_TEST=0

# Run tests for each dataset and error configuration
for dataset in "${DATASETS[@]}"; do
    echo -e "\n${CYAN}==================================================${NC}"
    echo -e "${CYAN}Testing dataset: ${dataset}${NC}"
    echo -e "${CYAN}==================================================${NC}"

    for config in "${ERROR_CONFIGS[@]}"; do
        # Extract error name and flags
        IFS=':' read -r error_name error_flags <<< "$config"
        
        # Update progress counter
        CURRENT_TEST=$((CURRENT_TEST + 1))
        PROGRESS=$(bc <<< "scale=2; ($CURRENT_TEST / $TOTAL_TESTS) * 100")
        
        # Format error name for filename
        error_file_name=$(echo "$error_name" | tr ' ' '_' | tr '+' 'plus')
        log_file_name="${RESULTS_DIR}/${dataset}_${error_file_name}.log"
        
        echo -e "[Progress: ${PROGRESS}%] Testing ${dataset} with ${error_name}"
        
        # Build the full parameter set
        full_params="${BASE_PARAMS} ${FUNCTIONS} --data_dir datasets_bc --function ${dataset} ${error_flags}"
        
        # Run the test
        run_test "${dataset} with ${error_name}" "${full_params}" "${log_file_name}"
    done
done

echo -e "\n${GREEN}All tests completed! Logs saved to ${RESULTS_DIR}${NC}"
echo -e "${GREEN}Total tests run: ${TOTAL_TESTS}${NC}" 