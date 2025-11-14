@echo off
REM Navigate to the directory of the batch file
cd /d "%~dp0"

REM (Optional) Activate your virtual environment first:
REM call "%USERPROFILE%\python\envs\myenv\Scripts\activate.bat"

REM Open cmd in the current project directory and run app.py
cmd /k "python app.py"