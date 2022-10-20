@REM 2022/10/04
@REM preprocess features for HuBERT-pitdyn
@ECHO OFF

SETLOCAL

SET PYTHON_BIN=python

SET VBANK=%1
SET WAVPATH=%2
IF "%VBANK%"=="" GOTO HELP
IF "%WAVPATH%"=="" GOTO HELP

ECHO ^>^> [0/2] making features for `%VBANK%` from "%WAVPATH%"
ECHO.

SET DATA_PATH=data\%VBANK%
SET OUT_PATH=out\%VBANK%
IF NOT EXIST %WAVPATH% (
  ECHO ^<^< [Error] wavpath "%WAVPATH%" does not exist!
  ECHO.
  EXIT /B -1
)

ECHO ^>^> [1/2] make workspace "%DATA_PATH%"
MKDIR %DATA_PATH%
ECHO ^>^> link "%WAVPATH%" to "%DATA_PATH%\wavs"
MKLINK /J %DATA_PATH%\wavs %WAVPATH%
ECHO.

ECHO ^>^> [2/2] prepare acoustics ^at "%DATA_PATH%\pits|dyns"
MKDIR %DATA_PATH%\pits
MKDIR %DATA_PATH%\dyns
%PYTHON_BIN% preprocess.py %VBANK% --pitdyn
ECHO.
IF ERRORLEVEL 1 EXIT /B -1

ECHO ^>^> done!

GOTO EOF

:HELP
ECHO Usage: %0 ^<vbank^> ^<wavpath^>
ECHO   vbank      voice bank name
ECHO   wavpath    folder path containing *.wav files
ECHO.

:EOF
