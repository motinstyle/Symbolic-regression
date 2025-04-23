# PowerShell script for testing sr_test.py with datasets_bc

# Colors for better readability
$Green = [ConsoleColor]::Green
$Cyan = [ConsoleColor]::Cyan
$Yellow = [ConsoleColor]::Yellow
$Red = [ConsoleColor]::Red

Write-Host "Starting Symbolic Regression test suite for datasets_bc..." -ForegroundColor $Green

# Base parameters for all tests
$BASE_PARAMS = "--epochs 100 --start_population_size 1000 --population_size 100 --runs 5 --save_results"
$FUNCTIONS = "--use_functions sum_,neg_,mult_,div_,sin_,cos_,atan_,pow2_,pow3_"

# Create results directory if it doesn't exist
$ResultsDir = "results"
if (!(Test-Path -Path $ResultsDir)) {
    New-Item -ItemType Directory -Path $ResultsDir -Force | Out-Null
    Write-Host "Created results directory: $ResultsDir" -ForegroundColor $Green
}

# Function to run tests with a header
function Run-Test {
    param (
        [string]$TestName,
        [string]$Params,
        [string]$LogFile
    )
    
    Write-Host "`n====================================" -ForegroundColor $Yellow
    Write-Host "Running test: $TestName" -ForegroundColor $Yellow
    Write-Host "Parameters: $Params" -ForegroundColor $Yellow
    Write-Host "Log file: $LogFile" -ForegroundColor $Yellow
    Write-Host "====================================" -ForegroundColor $Yellow
    
    # Execute the command with output redirection to log file
    $Command = "python sr/sr_test.py $Params *> $LogFile"
    Invoke-Expression $Command
    
    # Check if the command was successful
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Test completed successfully!" -ForegroundColor $Green
    } else {
        Write-Host "Test failed!" -ForegroundColor $Red
    }
}

# Get all datasets from the datasets_bc folder
$Datasets = Get-ChildItem "datasets_bc" -Filter "*.csv" | Select-Object -ExpandProperty Name | ForEach-Object { $_ -replace ".csv", "" }

# Define error types and their combinations
$ErrorConfigurations = @(
    @{Name = "Forward Error"; Flags = "--requires_forward_error"},
    @{Name = "Inverse Error"; Flags = "--requires_inv_error"},
    @{Name = "Absolute Inverse Error"; Flags = "--requires_abs_inv_error"},
    @{Name = "Spatial Absolute Inverse Error"; Flags = "--requires_spatial_abs_inv_error"},
    @{Name = "Forward + Inverse Error"; Flags = "--requires_forward_error --requires_inv_error"},
    @{Name = "Forward + Absolute Inverse Error"; Flags = "--requires_forward_error --requires_abs_inv_error"},
    @{Name = "Forward + Spatial Absolute Inverse Error"; Flags = "--requires_forward_error --requires_spatial_abs_inv_error"},
    @{Name = "Forward + Absolute + Spatial Absolute Error"; Flags = "--requires_forward_error --requires_abs_inv_error --requires_spatial_abs_inv_error"},
    @{Name = "All Error Types"; Flags = "--requires_forward_error --requires_inv_error --requires_abs_inv_error --requires_spatial_abs_inv_error"}
)

# Counter for total tests
$TotalTests = $Datasets.Count * $ErrorConfigurations.Count
$CurrentTest = 0

# Run tests for each dataset and error configuration
foreach ($Dataset in $Datasets) {
    Write-Host "`n==================================================" -ForegroundColor $Cyan
    Write-Host "Testing dataset: $Dataset" -ForegroundColor $Cyan
    Write-Host "==================================================" -ForegroundColor $Cyan

    foreach ($ErrorConfig in $ErrorConfigurations) {
        $CurrentTest++
        $Progress = [math]::Round(($CurrentTest / $TotalTests) * 100, 2)
        
        $ErrorName = $ErrorConfig.Name -replace " ", "_" -replace "\+", "plus"
        $LogFileName = "$ResultsDir/${Dataset}_${ErrorName}.log"
        
        Write-Host "[Progress: $Progress%] Testing $Dataset with $($ErrorConfig.Name)"
        
        # Build the full parameter set
        $FullParams = "$BASE_PARAMS $FUNCTIONS --data_dir datasets_bc --function $Dataset $($ErrorConfig.Flags)"
        
        # Run the test
        Run-Test -TestName "$Dataset with $($ErrorConfig.Name)" -Params $FullParams -LogFile $LogFileName
    }
}

Write-Host "`nAll tests completed! Logs saved to $ResultsDir" -ForegroundColor $Green
Write-Host "Total tests run: $TotalTests" -ForegroundColor $Green 