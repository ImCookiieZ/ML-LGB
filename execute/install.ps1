# Download Python 3.11 installer
Invoke-WebRequest -Uri https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe -OutFile python-3.11.0-amd64.exe

# Install Python 3.11
Start-Process -Wait -FilePath python-3.11.0-amd64.exe

# Check Python installation
python --version

# Check pip installation
pip --version

# Install required Python packages
pip install matplotlib seaborn lightgbm scikit-learn optunity joblib