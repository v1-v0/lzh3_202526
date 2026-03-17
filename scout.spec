# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['test.py'],  # Main script
    pathex=[],
    binaries=[],
    datas=[
        ('bacteria_configs', 'bacteria_configs'),  # Include config directory
        ('bacteria_configs.py', '.'),              # Include config module
    ],
    hiddenimports=[
        'cv2',
        'numpy',
        'pandas',
        'openpyxl',
        'openpyxl.chart',
        'openpyxl.chart.marker',
        'openpyxl.chart.series_factory',
        'openpyxl.drawing.image',
        'openpyxl.styles',
        'openpyxl.utils',
        'skimage',
        'skimage.registration',
        'scipy',
        'scipy.stats',
        'matplotlib',
        'matplotlib.pyplot',
        'seaborn',
        'tqdm',
        'PIL',
        'PIL.Image',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',  # Exclude if not needed
        'PyQt5',
        'PyQt6',
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
    name='MicrogelAnalysis',
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
    
)