@REM 2022/09/15
@REM Only perform data preprocess
@ECHO OFF

SETLOCAL

SET PYTHON_BIN=python

SET VBANK=%1
SET WAVPATH=%2
SET PITDYN=%3
IF "%VBANK%"=="" GOTO HELP
IF "%WAVPATH%"=="" GOTO HELP
IF "%PITDYN%"=="" GOTO HELP

ECHO ^>^> [0/2] making voicebank `%VBANK%` from "%WAVPATH%"
ECHO.

SET DATA_PATH=data\%VBANK%
SET OUT_PATH=out\%VBANK%
IF NOT EXIST %WAVPATH% (
  ECHO ^<^< [Error] wavpath "%WAVPATH%" does not exist!
  ECHO.
  EXIT /B -1
)

ECHO ^>^> [1/3] make workspace "%DATA_PATH%"
MKDIR %DATA_PATH%
ECHO ^>^> link "%WAVPATH%" to "%DATA_PATH%\wavs"
MKLINK /J %DATA_PATH%\wavs %WAVPATH%
ECHO.

ECHO ^>^> [2/3] prepare mels ^at "%DATA_PATH%\mels"
MKDIR %DATA_PATH%\mels
%PYTHON_BIN% preprocess.py %VBANK% --mel
ECHO.
IF ERRORLEVEL 1 EXIT /B -1

ECHO ^>^> [3/3] prepare hubert-pitdyn units ^at "%DATA_PATH%\units-pitdyn"
MKDIR %DATA_PATH%\units-pitdyn
%PYTHON_BIN% preprocess.py %VBANK% --encode %PITDYN%
ECHO.

IF ERRORLEVEL 1 EXIT /B -1

ECHO ^>^> done!

GOTO EOF


:HELP
ECHO Usage: %0 ^<vbank^> ^<wavpath^> ^<pitdyn^>
ECHO   vbank      voice bank name
ECHO   wavpath    folder path containing *.wav files
ECHO   pitdyn     dataset name for your locally trained hubert-pitdyn
ECHO.

:EOF
