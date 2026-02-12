# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['dev_viewer.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'PIL._tkinter_finder',
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.scrolledtext',
        'pandas',
        'matplotlib',
        'matplotlib.backends.backend_tkagg',
        'numpy',
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'reportlab',
        'reportlab.lib.pagesizes',
        'reportlab.platypus',
        'reportlab.lib.styles',
        'reportlab.lib.units',
        'reportlab.lib.colors',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'pytest',
        'scipy',
        'IPython',
        'jupyter',
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
    name='Particle-Viewer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='viewer_icon.ico',  # Optional: add your icon file
    version='version_info.txt',  # Optional: version info
)