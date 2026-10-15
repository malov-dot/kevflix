@echo off
set "KEVFLIX_FIREWALL_HELPER=%~dp0Allow-KevFLIX-LAN.ps1"
powershell.exe -NoProfile -Command "$p=$env:KEVFLIX_FIREWALL_HELPER; $quoted='\"'+$p+'\"'; Start-Process powershell.exe -Verb RunAs -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-NoExit','-File',$quoted)"
