#!/bin/bash

# Colors for better readability
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Symbolic Regression test suite...${NC}"

# Base parameters for all tests
BASE_PARAMS="--epochs 50 --start_population_size 100 --population_size 10 --runs 5 --save_results"

# Function to run tests with a header
run_test() {
    local test_name=$1
    local params=$2
    
    echo -e "\n${YELLOW}====================================${NC}"
    echo -e "${YELLOW}Running test: ${test_name}${NC}"
    echo -e "${YELLOW}Parameters: ${params}${NC}"
    echo -e "${YELLOW}====================================${NC}"
    
    # Execute the command
    python to_git/sr/sr_test.py ${params}
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Test completed successfully!${NC}"
    else
        echo -e "${RED}Test failed!${NC}"
    fi
}

# Section 1: Test individual error types
echo -e "\n${CYAN}==================================================${NC}"
echo -e "${CYAN}SECTION 1: Testing individual error types${NC}"
echo -e "${CYAN}==================================================${NC}"

run_test "Forward Error" "${BASE_PARAMS} --requires_forward_error"
run_test "Inverse Error" "${BASE_PARAMS} --requires_inv_error"
run_test "Absolute Inverse Error" "${BASE_PARAMS} --requires_abs_inv_error"
run_test "Spatial Absolute Inverse Error" "${BASE_PARAMS} --requires_spatial_abs_inv_error"

# Section 2: Test forward error combined with other types
echo -e "\n${CYAN}==================================================${NC}"
echo -e "${CYAN}SECTION 2: Testing forward error with other types${NC}"
echo -e "${CYAN}==================================================${NC}"

run_test "Forward + Inverse Error" "${BASE_PARAMS} --requires_forward_error --requires_inv_error"
run_test "Forward + Absolute Inverse Error" "${BASE_PARAMS} --requires_forward_error --requires_abs_inv_error"
run_test "Forward + Spatial Absolute Inverse Error" "${BASE_PARAMS} --requires_forward_error --requires_spatial_abs_inv_error"
run_test "All Error Types" "${BASE_PARAMS} --requires_forward_error --requires_inv_error --requires_abs_inv_error --requires_spatial_abs_inv_error"

# Section 3: Test different datasets
echo -e "\n${CYAN}==================================================${NC}"
echo -e "${CYAN}SECTION 3: Testing different datasets${NC}"
echo -e "${CYAN}==================================================${NC}"

# Default dataset with all error types
run_test "Default Dataset" "${BASE_PARAMS} --requires_forward_error --data_dir ../new_datasets"

# Test datasets with all error types
run_test "Test Dataset" "${BASE_PARAMS} --requires_forward_error --data_dir ../datasets_test"

# Section 4: Test specific functions from datasets
echo -e "\n${CYAN}==================================================${NC}"
echo -e "${CYAN}SECTION 4: Testing specific functions${NC}"
echo -e "${CYAN}==================================================${NC}"

run_test "Function: x*sin(x)" "${BASE_PARAMS} --requires_forward_error --data_dir ../new_datasets --function x_sin_x"
run_test "Function: x^2+3*x+5" "${BASE_PARAMS} --requires_forward_error --data_dir ../new_datasets --function x2_3x_5"

# Section 5: Test with different function subsets
echo -e "\n${CYAN}==================================================${NC}"
echo -e "${CYAN}SECTION 5: Testing function subsets${NC}"
echo -e "${CYAN}==================================================${NC}"

run_test "Only Arithmetic Functions" "${BASE_PARAMS} --requires_forward_error --use_functions sum_,sub_,mult_,div_"
run_test "Only Trigonometric Functions" "${BASE_PARAMS} --requires_forward_error --use_functions sin_,cos_,tan_"

# Section 6: Test with gradient enabled
echo -e "\n${CYAN}==================================================${NC}"
echo -e "${CYAN}SECTION 6: Testing with gradient computation${NC}"
echo -e "${CYAN}==================================================${NC}"

run_test "With Gradient" "${BASE_PARAMS} --requires_forward_error --requires_grad"

echo -e "\n${GREEN}All tests completed!${NC}"
