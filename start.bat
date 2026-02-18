@echo off
title LFM Speech-to-Type
echo ================================================
echo   LFM Speech-to-Type  (Liquid AI LFM2.5-Audio)
echo ================================================
echo.

set PYTHON=python

REM Install dependencies if needed
%PYTHON% -c "import keyboard" 2>nul || (
    echo Installing Python dependencies...
    %PYTHON% -m pip install -r requirements.txt
    echo.
)

echo Starting...  Edit config.ini to change hotkey or settings.
echo.
%PYTHON% lfm_speech_to_type.py %*
pause
