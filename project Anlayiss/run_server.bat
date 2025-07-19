@echo off
echo Installing required packages...
pip install -r requirements.txt

echo Starting server...
python project_analyzer.py

    pause
