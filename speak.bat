@echo off
rem Read text aloud in your cloned voice.  Usage: speak.bat my_voice.wav "Text to read" [options]
setlocal
set "HERE=%~dp0"
set "PYTHONPATH=%HERE%;%PYTHONPATH%"
"%HERE%.venv\Scripts\python.exe" -m vclone %*
exit /b %ERRORLEVEL%
