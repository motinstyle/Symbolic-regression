# PowerShell script for testing sr_test.py with various parameters

# Colors for better readability
$Green = [ConsoleColor]::Green
$Cyan = [ConsoleColor]::Cyan
$Yellow = [ConsoleColor]::Yellow
$Red = [ConsoleColor]::Red

Write-Host "Starting Symbolic Regression test suite..." -ForegroundColor $Green

# Base parameters for all tests
$BASE_PARAMS = "--epochs 20 --start_population_size 100 --population_size 10 --runs 2 --save_results"

# Function to run tests with a header
function Run-Test {
    param (
        [string]$TestName,
        [string]$Params
    )
    
    Write-Host "`n====================================" -ForegroundColor $Yellow
    Write-Host "Running test: $TestName" -ForegroundColor $Yellow
    Write-Host "Parameters: $Params" -ForegroundColor $Yellow
    Write-Host "====================================" -ForegroundColor $Yellow
    
    # Execute the command
    # $Command = "python to_git/sr/sr_test.py $Params"
    $Command = "python sr/sr_test.py $Params"
    Invoke-Expression $Command
    
    # Check if the command was successful
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Test completed successfully!" -ForegroundColor $Green
    } else {
        Write-Host "Test failed!" -ForegroundColor $Red
    }
}

# Section 1: Test individual error types
Write-Host "`n==================================================" -ForegroundColor $Cyan
Write-Host "SECTION 1: Testing individual error types" -ForegroundColor $Cyan
Write-Host "==================================================" -ForegroundColor $Cyan

Run-Test -TestName "Forward Error" -Params "$BASE_PARAMS --requires_forward_error *> results/forward_error.log"
Run-Test -TestName "Inverse Error" -Params "$BASE_PARAMS --requires_inv_error *> results/inverse_error.log"
Run-Test -TestName "Absolute Inverse Error" -Params "$BASE_PARAMS --requires_abs_inv_error *> results/absolute_inverse_error.log"
Run-Test -TestName "Spatial Absolute Inverse Error" -Params "$BASE_PARAMS --requires_spatial_abs_inv_error *> results/spatial_absolute_inverse_error.log"

# Section 2: Test forward error combined with other types
Write-Host "`n==================================================" -ForegroundColor $Cyan
Write-Host "SECTION 2: Testing forward error with other types" -ForegroundColor $Cyan
Write-Host "==================================================" -ForegroundColor $Cyan

Run-Test -TestName "Forward + Inverse Error" -Params "$BASE_PARAMS --requires_forward_error --requires_inv_error *> results/forward_inverse_error.log"
Run-Test -TestName "Forward + Absolute Inverse Error" -Params "$BASE_PARAMS --requires_forward_error --requires_abs_inv_error *> results/forward_absolute_inverse_error.log"
Run-Test -TestName "Forward + Spatial Absolute Inverse Error" -Params "$BASE_PARAMS --requires_forward_error --requires_spatial_abs_inv_error *> results/forward_spatial_absolute_inverse_error.log"
Run-Test -TestName "All Error Types" -Params "$BASE_PARAMS --requires_forward_error --requires_inv_error --requires_abs_inv_error --requires_spatial_abs_inv_error *> results/all_error_types.log"

# Section 3: Test different datasets
Write-Host "`n==================================================" -ForegroundColor $Cyan
Write-Host "SECTION 3: Testing different datasets" -ForegroundColor $Cyan
Write-Host "==================================================" -ForegroundColor $Cyan

# Default dataset with all error types
Run-Test -TestName "Default Dataset" -Params "$BASE_PARAMS --requires_forward_error --data_dir ../new_datasets *> results/default_dataset.log"

# Test datasets with all error types
Run-Test -TestName "Test Dataset" -Params "$BASE_PARAMS --requires_forward_error --data_dir ../datasets_test *> results/test_dataset.log"

# Section 4: Test specific functions from datasets
Write-Host "`n==================================================" -ForegroundColor $Cyan
Write-Host "SECTION 4: Testing specific functions" -ForegroundColor $Cyan
Write-Host "==================================================" -ForegroundColor $Cyan

Run-Test -TestName "Function: x*sin(x)" -Params "$BASE_PARAMS --requires_forward_error --data_dir ../new_datasets --function x_sin_x *> results/x_sin_x.log"
Run-Test -TestName "Function: x^2+3*x+5" -Params "$BASE_PARAMS --requires_forward_error --data_dir ../new_datasets --function x2_3x_5 *> results/x2_3x_5.log"

# Section 5: Test with different function subsets
Write-Host "`n==================================================" -ForegroundColor $Cyan
Write-Host "SECTION 5: Testing function subsets" -ForegroundColor $Cyan
Write-Host "==================================================" -ForegroundColor $Cyan

Run-Test -TestName "Only Arithmetic Functions" -Params "$BASE_PARAMS --requires_forward_error --use_functions sum_,sub_,mult_,div_ *> results/only_arithmetic_functions.log"
Run-Test -TestName "Only Trigonometric Functions" -Params "$BASE_PARAMS --requires_forward_error --use_functions sin_,cos_,tan_ *> results/only_trigonometric_functions.log"

# Section 6: Test with gradient enabled
Write-Host "`n==================================================" -ForegroundColor $Cyan
Write-Host "SECTION 6: Testing with gradient computation" -ForegroundColor $Cyan
Write-Host "==================================================" -ForegroundColor $Cyan

Run-Test -TestName "With Gradient" -Params "$BASE_PARAMS --requires_forward_error --requires_grad *> results/with_gradient.log"

Write-Host "`nAll tests completed!" -ForegroundColor $Green 