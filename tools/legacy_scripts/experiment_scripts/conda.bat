@echo off
setlocal ENABLEEXTENSIONS

echo ==========================================================
echo  Auto-Setup Conda Env Vars for CMD
echo  - Detect CONDA_HOME
echo  - Set user env: CONDA_HOME and Path
echo  - conda init cmd.exe
echo ==========================================================
echo.

REM --------------- Step 1: Try to find conda on PATH ---------------
set "CONDA_HOME="
set "FOUND_CONDA="

for /f "usebackq delims=" %%I in (`where conda 2^>nul`) do (
  if not defined FOUND_CONDA (
    set "FOUND_CONDA=%%~fI"
  )
)

if defined FOUND_CONDA (
  REM example: ...\anaconda3\condabin\conda.bat  OR  ...\anaconda3\Scripts\conda.exe
  for %%I in ("%FOUND_CONDA%") do set "FOUND_DIR=%%~dpI"
  for %%I in ("%FOUND_DIR%\..") do set "CONDA_HOME=%%~fI"
)

REM --------------- Step 2: Probe common install locations if not found ---------------
if not defined CONDA_HOME (
  for %%D in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "C:\ProgramData\Anaconda3" "C:\Anaconda3" "D:\anaconda3" "D:\miniconda3") do (
    if exist "%%~D\Scripts\conda.exe" (
      set "CONDA_HOME=%%~D"
      goto :FOUND
    )
  )
)

:FOUND
if not defined CONDA_HOME (
  echo [ERROR] 未找到 Conda 安装目录。请确认已安装 Anaconda/Miniconda。
  echo         你也可以在腳本開頭手動設置 CONDA_HOME 後再運行。
  exit /b 1
)

REM 规范化去掉末尾反斜杠
for %%I in ("%CONDA_HOME%") do set "CONDA_HOME=%%~fI"

echo [INFO] CONDA_HOME = %CONDA_HOME%
if not exist "%CONDA_HOME%\Scripts\conda.exe" (
  echo [ERROR] 路徑看起來不對：未找到 "%CONDA_HOME%\Scripts\conda.exe"
  exit /b 2
)

REM --------------- Step 3: Set user env vars ---------------
echo.
echo [STEP] 寫入用戶環境變量（不影響系統變量）
echo        - CONDA_HOME
echo        - Path 追加三條：^
%CONDA_HOME% ^| %CONDA_HOME%\Scripts ^| %CONDA_HOME%\Library\bin
echo.

REM 設置 CONDA_HOME（用戶變量）
setx CONDA_HOME "%CONDA_HOME%" >nul

REM 追加到 Path（用戶變量）。注意：可能會有重複，屬於無害。
setx PATH "%PATH%;%CONDA_HOME%;%CONDA_HOME%\Scripts;%CONDA_HOME%\Library\bin" >nul

REM --------------- Step 4: conda init for cmd.exe ---------------
echo.
echo [STEP] 初始化 conda 對 CMD 的支持：conda init cmd.exe
call "%CONDA_HOME%\Scripts\conda.exe" init cmd.exe

echo.
echo ==========================================================
echo  ✅ 已完成配置（用戶變量）。
echo  請關閉並重新打開 CMD，然後執行：
echo      conda --version
echo      conda activate base
echo ==========================================================
echo.
pause
endlocal
