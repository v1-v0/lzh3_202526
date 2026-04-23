# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for main-scout.py
# Build command: pyinstaller --clean main-scout.spec

block_cipher = None

a = Analysis(
    ['main-scout.py'],
    pathex=[],
    binaries=[],
    
    # ── Data files bundled into the executable ────────────────────────────────
    datas=[
        # Bundles the directory of JSON files
        ('bacteria_configs', 'bacteria_configs'),
        
        # Uncomment the line below if you want config.ini bundled inside the exe
        # ('config.ini', '.'),
    ],

    # ── Hidden imports ────────────────────────────────────────────────────────
    hiddenimports=[
        # Standard library
        'hashlib', 'csv', 'json', 'logging', 'xml.etree.ElementTree',

        # Data Science & Vision
        'numpy', 'numpy.core._methods', 'numpy.lib.stride_tricks', 'numpy.random',
        'pandas', 'pandas.io.formats.excel', 'pandas.io.excel._openpyxl',
        'scipy', 'scipy.stats', 'scipy.special', 'scipy.linalg', 'scipy.sparse',
        'cv2', 

        # Plotting
        'matplotlib', 'matplotlib.pyplot', 'matplotlib.backends.backend_agg',
        'matplotlib.backends.backend_pdf', 'seaborn',

        # Excel & Imaging
        'openpyxl', 'PIL', 'tqdm',
        
        # PDF & Report Generation (imported in try/except)
        'reportlab', 'pypdf', 'PyPDF2',

        # ── Local application modules (imported dynamically) ──────────────────
        'bacteria_configs',     
        'bacteria_registry',    
        'run_profiler',         # Added to ensure the profiler is bundled
    ],

    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],

    # ── Packages to exclude to keep the bundle small ──────────────────────────
    excludes=[
        'tkinter', '_tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'wx', 'gi',
        'IPython', 'jupyter', 'notebook', 'nbformat', 'nbconvert', 'pytest',
        'sphinx', 'docutils', 'jedi', 'pyarrow', 'tables', 'h5py', 'numba',
        'sympy', 'statsmodels', 'sklearn', 'tensorflow', 'torch', 'keras',
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
    name='MainScout',
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
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)