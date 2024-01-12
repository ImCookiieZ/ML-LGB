# Get the directory where the PowerShell script is located
$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Definition

$pythonScript = "DDA-LGB-ML.py"

# Build the full path to the Python script
$pythonScriptPath = Join-Path $scriptDirectory $pythonScript

# Execute the Python script
python $pythonScriptPath