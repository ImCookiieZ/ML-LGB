# Get the directory where the PowerShell script is located
$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Definition

$pythonScript = "DDA-LGB-ML.py"

# Build the full path to the Python script
$pythonScriptPath = Join-Path $scriptDirectory $pythonScript

# Execute the Python script
$process = Start-Process -FilePath "python" -ArgumentList $pythonScriptPath -PassThru -Wait

# Check the exit code of the Python script
if ($process.ExitCode -eq 0) {
    Write-Host "Python script completed successfully. Exit code: $($process.ExitCode)"
    exit 0
} else {
    Write-Host "Python script encountered an error. Exit code: $($process.ExitCode)"
    exit $process.ExitCode
}