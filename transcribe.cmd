@echo off
rem Console transcription:  transcribe.cmd my_audio.m4a [--beam 1] [--device cpu]
"%~dp0.venv\Scripts\python.exe" "%~dp0transcribe_ivrit.py" %*
