@echo off
REM Quick sync script - runs the PowerShell script
powershell -ExecutionPolicy Bypass -File "%~dp0sync-to-box.ps1"
pause
