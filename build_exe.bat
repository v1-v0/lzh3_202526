@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo    PARTICLE-SCOUT VIEWER - PyInstaller Build Script
echo    Version 2.2 - Single EXE Build
echo ================================================================
echo.

:: Check if Python is available
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and try again.
    pause
    exit /b 1
)

echo [Step 1/6] Checking Python installation...
python --version
echo.

:: Check if dev_viewer.py exists
if not exist "dev_viewer.py" (
    echo ERROR: dev_viewer.py not found in current directory
    echo Please run this script from the folder containing dev_viewer.py
    pause
    exit /b 1
)

echo [Step 2/6] Cleaning previous builds...
if exist build (
    echo    Removing build folder...
    rmdir /s /q build
)
if exist dist (
    echo    Removing dist folder...
    rmdir /s /q dist
)
if exist *.spec (
    echo    Removing old spec files...
    del /q *.spec
)
echo    Clean complete.
echo.

echo [Step 3/6] Installing/Updating required packages...
echo    This may take a few minutes...
pip install --upgrade pip >nul 2>&1
pip install --upgrade pyinstaller pillow pandas matplotlib reportlab
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo    Dependencies installed successfully.
echo.

echo [Step 4/6] Building executable with PyInstaller...
echo    Creating single EXE file (this may take 5-10 minutes)...
echo.

pyinstaller --noconfirm ^
    --onefile ^
    --windowed ^
    --name "ParticleScoutViewer" ^
    --add-data "README.md;." ^
    --hidden-import "PIL._tkinter_finder" ^
    --hidden-import "PIL.Image" ^
    --hidden-import "PIL.ImageTk" ^
    --hidden-import "pandas" ^
    --hidden-import "matplotlib" ^
    --hidden-import "matplotlib.backends.backend_tkagg" ^
    --hidden-import "matplotlib.figure" ^
    --hidden-import "reportlab" ^
    --hidden-import "reportlab.lib.pagesizes" ^
    --hidden-import "reportlab.platypus" ^
    --hidden-import "reportlab.lib.styles" ^
    --hidden-import "reportlab.lib.units" ^
    --hidden-import "reportlab.lib.colors" ^
    --collect-all "matplotlib" ^
    --collect-all "reportlab" ^
    --exclude-module "pytest" ^
    --exclude-module "scipy" ^
    --exclude-module "IPython" ^
    --exclude-module "jupyter" ^
    --exclude-module "notebook" ^
    --exclude-module "sphinx" ^
    "dev_viewer.py"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: PyInstaller build failed!
    echo Check the error messages above.
    pause
    exit /b 1
)

echo.
echo [Step 5/6] Verifying build...

if exist "dist\ParticleScoutViewer.exe" (
    echo    SUCCESS: Executable created!
    echo.
    echo    Location: dist\ParticleScoutViewer.exe
    for %%I in ("dist\ParticleScoutViewer.exe") do (
        set size=%%~zI
        set /a size_mb=!size! / 1048576
        echo    Size: !size_mb! MB
    )
) else (
    echo    ERROR: Executable not found!
    pause
    exit /b 1
)

echo.
echo [Step 6/6] Creating distribution package...

:: Create distribution folder
set timestamp=%date:~-4%%date:~4,2%%date:~7,2%
set dist_folder=ParticleScoutViewer_v2.2_%timestamp%
if exist "%dist_folder%" rmdir /s /q "%dist_folder%"
mkdir "%dist_folder%"

:: Copy executable
copy "dist\ParticleScoutViewer.exe" "%dist_folder%\" >nul
echo    Copied executable to %dist_folder%

:: Create README
(
echo PARTICLE-SCOUT CLINICAL RESULTS VIEWER
echo Version 2.2 - Single Executable
echo ================================================================
echo.
echo QUICK START:
echo 1. Double-click ParticleScoutViewer.exe
echo 2. Click "Load Results" ^(or press Ctrl+O^)
echo 3. Select your output folder
echo 4. Browse groups and view results
echo.
echo SYSTEM REQUIREMENTS:
echo - Windows 10/11 ^(64-bit^)
echo - 4GB RAM minimum
echo - 100MB free disk space
echo.
echo FEATURES:
echo - Batch processing ^(G+ and G-^)
echo - Multi-sample navigation
echo - Processing pipeline visualization
echo - Clinical classification
echo - Data export ^(CSV/PDF^)
echo - Control group support
echo.
echo KEYBOARD SHORTCUTS:
echo - Ctrl+O: Open results folder
echo - Ctrl+R: Open recent folder
echo - F5: Refresh view
echo - Ctrl+Q: Quit
echo.
echo For detailed help, use Help menu in the application.
echo.
echo Copyright ^(C^) 2026 - All Rights Reserved
) > "%dist_folder%\README.txt"

echo    Created README.txt

:: Create ZIP if possible
where 7z >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo    Creating ZIP archive with 7-Zip...
    7z a -tzip "%dist_folder%.zip" "%dist_folder%\*" >nul
    if exist "%dist_folder%.zip" (
        echo    Created: %dist_folder%.zip
    )
) else (
    echo    ^(7-Zip not found - skipping ZIP creation^)
)

echo.
echo ================================================================
echo    BUILD COMPLETED SUCCESSFULLY!
echo ================================================================
echo.
echo Executable location:
echo    %CD%\dist\ParticleScoutViewer.exe
echo.
echo Distribution package:
echo    %CD%\%dist_folder%\
echo.
echo You can now:
echo 1. Test: dist\ParticleScoutViewer.exe
echo 2. Distribute: %dist_folder%\ParticleScoutViewer.exe
echo.
echo Press any key to exit...
pause >nul