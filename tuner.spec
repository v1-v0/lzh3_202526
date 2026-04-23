# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for tuner.py
# Build command: pyinstaller --clean tuner.spec

block_cipher = None

a = Analysis(
    ['tuner.py'],
    pathex=[],
    binaries=[],
    
    # ── Data files bundled into the executable ────────────────────────────────
    datas=[
        # Bundle the directory of JSON files so the tuner can read/edit them
        ('bacteria_configs', 'bacteria_configs'),
    ],

    # ── Hidden imports ────────────────────────────────────────────────────────
    hiddenimports=[
        # GUI & Plotting
        'tkinter',
        'matplotlib',
        'matplotlib.backends.backend_tkagg',
        
        # Third-party libraries used in tuner.py
        'astor',
        'arrow',
        'zmq',
        'cv2',
        'numpy',

        # Local application modules
        'bacteria_configs',
        'bacteria_registry',
    ],

    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],

    # ── Packages to exclude to keep the bundle small ──────────────────────────
    # We exclude alternative GUI frameworks and heavy ML libraries
    excludes=[
        'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'wx', 'gi',
        'IPython', 'jupyter', 'notebook', 'pytest', 'sphinx',
        'tensorflow', 'torch', 'keras', 'sklearn', 'tables', 'h5py'
    ],

    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='PathogenTuner',           # Output executable name
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[
        'vcruntime140.dll',
        'python3*.dll',
        'cv2*.pyd',
    ],
    runtime_tmpdir=None,
    
    # ── CONSOLE SETTING ───────────────────────────────────────────────────────
    # Set to False to hide the black terminal window behind the GUI.
    # Set to True if you want to see the print() statements for debugging.
    console=False,                  
    
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)