@echo off
REM Sports Broadcast Detector Launcher
REM Double-click this file to start the detector
REM You can edit this file to specify a different sport (e.g., --sport nba)

cd /d "%~dp0"
python detector.py
pause

