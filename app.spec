# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# List all your dependencies
hiddenimports = [
    'numpy',
    'pandas',
    'cv2',
    'scipy',
    'scipy.stats',
    'matplotlib',
    'seaborn',
    'openpyxl',
    'skimage',
    'skimage.registration',
    'tqdm',
    'arrow',
    'pyparsing',
    'bacteria_configs',
]

# Data files to include
datas = [
    ('bacteria_configs', 'bacteria_configs'),  # Include config directory
]

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyt = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyt,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='particle_scout',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to False for GUI-only mode
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add 'icon.ico' if you have one
)