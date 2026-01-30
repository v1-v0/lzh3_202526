@echo off
echo ========================================
echo Building ParticleScout Suite
echo ========================================
echo.

REM Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist *.spec del *.spec

REM Build Analyzer (folder-based)
echo [1/3] Building Analyzer...
echo.
pyinstaller --onedir --console --name ParticleScout-Analyzer ^
    --hidden-import numpy ^
    --hidden-import pandas ^
    --hidden-import cv2 ^
    --hidden-import scipy.stats ^
    --hidden-import matplotlib ^
    --hidden-import seaborn ^
    --hidden-import openpyxl ^
    --hidden-import skimage.registration ^
    --collect-data matplotlib ^
    test.py

if errorlevel 1 (
    echo Error building Analyzer!
    pause
    exit /b 1
)

REM Build Viewer (folder-based)
echo.
echo [2/3] Building Viewer...
echo.
pyinstaller --onedir --windowed --name ParticleScout-Viewer ^
    --hidden-import PIL._tkinter_finder ^
    --hidden-import PIL.Image ^
    --hidden-import PIL.ImageTk ^
    launcher.py

if errorlevel 1 (
    echo Error building Viewer!
    pause
    exit /b 1
)

REM Merge into one folder
echo.
echo [3/3] Merging into ParticleScout-Suite...
echo.

REM Copy Viewer exe into Analyzer folder
copy /Y "dist\ParticleScout-Viewer\ParticleScout-Viewer.exe" "dist\ParticleScout-Analyzer\"

REM Copy unique Viewer dependencies (if any)
xcopy /E /I /Y /Q "dist\ParticleScout-Viewer\_internal\*" "dist\ParticleScout-Analyzer\_internal\" 2>nul

REM Rename to Suite
if exist "dist\ParticleScout-Suite" rmdir /S /Q "dist\ParticleScout-Suite"
move "dist\ParticleScout-Analyzer" "dist\ParticleScout-Suite"

REM Clean up
rmdir /S /Q "dist\ParticleScout-Viewer"

echo.
echo ========================================
echo Build Complete!
echo ========================================
echo.
echo Location: dist\ParticleScout-Suite\
echo.
echo Executables:
dir /B "dist\ParticleScout-Suite\*.exe"
echo.
echo Total size:
powershell -Command "'{0:N2} MB' -f ((Get-ChildItem -Path 'dist\ParticleScout-Suite' -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB)"
echo.
echo Distribution folder is ready!
echo.
pause