# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for main-scout.py
# Build command (from project root):
#   pyinstaller scout.spec
#
# Required project layout:
#   project_root/
#     main-scout.py
#     bacteria_configs.py          <- SegmentationConfig dataclass
#     bacteria_registry.py         <- whitelist registry (optional but recommended)
#     bacteria_configs/            <- directory of *.json configuration files
#       default.json
#       klebsiella_pneumoniae.json
#       proteus_mirabilis.json
#       ... etc.
#     source/                      <- runtime input; NOT bundled
#     outputs/                     <- created at runtime; NOT bundled
#
# NOTE: The 'source/' directory is intentionally excluded from the bundle.
#       The executable expects it to exist next to the .exe at runtime.
#
# NOTE: bacteria_registry.py is optional. If it does not exist in your
#       project root, remove its two entries (marked OPTIONAL below).
#       main-scout.py handles the missing module gracefully via try/except.

block_cipher = None

a = Analysis(
    # ── Entry point ──────────────────────────────────────────────────────────
    ['main-scout.py'],

    pathex=[],
    binaries=[],

    # ── Data files bundled into the executable ────────────────────────────────
    # Format: (source_path, dest_folder_inside_bundle)
    datas=[
        # JSON bacteria configuration files (entire directory)
        ('bacteria_configs', 'bacteria_configs'),

        # SegmentationConfig dataclass — imported as 'bacteria_configs' module
        ('bacteria_configs.py', '.'),

        # Whitelist registry — imported in _load_multi_scan_whitelist()
        # OPTIONAL: remove this line if bacteria_registry.py does not exist
        ('bacteria_registry.py', '.'),
    ],

    # ── Hidden imports ────────────────────────────────────────────────────────
    # PyInstaller's static analysis misses these because they are imported
    # inside functions, loaded via importlib, or are C-extension sub-modules.
    hiddenimports=[

        # ── Standard library (dynamic imports inside functions) ───────────────
        'hashlib',          # get_cache_key() uses hashlib.md5 inside the function
        'csv',
        'json',
        'logging',
        'xml.etree.ElementTree',

        # ── NumPy ─────────────────────────────────────────────────────────────
        'numpy',
        'numpy.core',
        'numpy.core._multiarray_umath',
        'numpy.core._methods',
        'numpy.lib',
        'numpy.lib.stride_tricks',
        'numpy.random',             # used in _draw_chart_on_axis jitter

        # ── Pandas ────────────────────────────────────────────────────────────
        'pandas',
        'pandas._libs',
        'pandas._libs.tslibs',
        'pandas.io.formats.excel',  # required by pd.ExcelWriter engine dispatch
        'pandas.io.excel._openpyxl',# openpyxl engine for ExcelWriter
        'pandas.core.dtypes.cast',

        # ── SciPy ─────────────────────────────────────────────────────────────
        'scipy',
        'scipy.stats',
        'scipy.stats._stats_py',    # ttest_ind, sem implementation
        'scipy.stats._continuous_distns',  # t.ppf
        'scipy.special',            # dependency of scipy.stats
        'scipy.special._ufuncs',
        'scipy.linalg',
        'scipy.linalg.blas',
        'scipy.linalg.lapack',
        'scipy.fft',                # indirect dependency via phase correlation
        'scipy._lib',
        'scipy._lib.messagestream', # frequently missed by analysis
        'scipy._lib._util',
        'scipy.sparse',             # indirect skimage dependency

        # ── OpenCV ────────────────────────────────────────────────────────────
        'cv2',

        # ── scikit-image ──────────────────────────────────────────────────────
        'skimage',
        'skimage.registration',
        'skimage.registration._phase_cross_correlation',  # actual implementation
        'skimage._shared',
        'skimage._shared.utils',
        'skimage._shared.transform',
        'skimage._shared.fft',
        'skimage.util',
        'skimage.util.dtype',

        # ── Matplotlib ────────────────────────────────────────────────────────
        'matplotlib',
        'matplotlib.pyplot',
        'matplotlib.axes',
        'matplotlib.axes._axes',
        'matplotlib.figure',
        'matplotlib.patches',           # mpatches used in _draw_decision_heatmap_on_axis
        'matplotlib.lines',             # Line2D used in forest plot legend
        'matplotlib.collections',
        'matplotlib.ticker',
        'matplotlib.colors',
        'matplotlib.font_manager',
        'matplotlib.backends',
        'matplotlib.backends.backend_agg',      # matplotlib.use("Agg") at module level
        'matplotlib.backends.backend_pdf',      # PdfPages imported inside function

        # ── Seaborn ───────────────────────────────────────────────────────────
        'seaborn',
        'seaborn.axisgrid',         # FacetGrid / barplot internals
        'seaborn.utils',
        'seaborn._statistics',      # errorbar / estimator internals
        'seaborn.categorical',      # barplot, stripplot
        'seaborn.relational',
        'seaborn.distributions',
        'seaborn.palettes',         # color_palette used in comparison plots

        # ── openpyxl ──────────────────────────────────────────────────────────
        'openpyxl',
        'openpyxl.workbook',
        'openpyxl.worksheet',
        'openpyxl.worksheet.worksheet',
        'openpyxl.chart',
        'openpyxl.chart.bar_chart',
        'openpyxl.chart.scatter_chart',     # ScatterChart used in Ratios sheet
        'openpyxl.chart.reference',         # Reference
        'openpyxl.chart.marker',            # Marker
        'openpyxl.chart.series_factory',    # SeriesFactory
        'openpyxl.chart.data_source',
        'openpyxl.drawing',
        'openpyxl.drawing.image',           # XLImage used to embed PNG plots
        'openpyxl.drawing.spreadsheet_drawing',
        'openpyxl.styles',
        'openpyxl.styles.fills',            # PatternFill
        'openpyxl.styles.fonts',            # Font
        'openpyxl.styles.alignment',        # Alignment
        'openpyxl.styles.borders',          # Border, Side
        'openpyxl.styles.colors',
        'openpyxl.styles.numbers',
        'openpyxl.styles.named_styles',
        'openpyxl.utils',
        'openpyxl.utils.dataframe',         # used internally by ExcelWriter
        'openpyxl.utils.cell',

        # ── Pillow ────────────────────────────────────────────────────────────
        'PIL',
        'PIL.Image',
        'PIL.ImageFile',
        'PIL.PngImagePlugin',               # PNG read/write for XLImage
        'PIL.JpegImagePlugin',

        # ── tqdm ──────────────────────────────────────────────────────────────
        'tqdm',
        'tqdm.auto',
        'tqdm.std',

        # ── Local application modules ─────────────────────────────────────────
        'bacteria_configs',     # SegmentationConfig dataclass
        'bacteria_registry',    # OPTIONAL: whitelist registry
                                # safe to leave here even if the file is absent;
                                # the import error is caught in _load_multi_scan_whitelist
    ],

    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],

    # ── Packages to exclude to keep the bundle small ──────────────────────────
    excludes=[
        'tkinter',
        '_tkinter',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
        'wx',
        'gi',           # GTK
        'IPython',
        'jupyter',
        'notebook',
        'nbformat',
        'nbconvert',
        'pytest',
        'sphinx',
        'docutils',
        'jedi',
        'pyarrow',
        'tables',       # PyTables / HDF5
        'h5py',
        'numba',
        'sympy',
        'statsmodels',
        'sklearn',      # scikit-learn — not used
        'tensorflow',
        'torch',
        'keras',
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
    name='MicrogelScout',           # output executable name
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[
        # UPX can corrupt some DLLs — exclude these if the exe crashes on launch
        'vcruntime140.dll',
        'python3*.dll',
        'cv2*.pyd',
    ],
    runtime_tmpdir=None,
    console=True,                   # keep True — this is a CLI pipeline
                                    # set to False only if you add a GUI wrapper
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,               # None = native arch; set 'x86_64' to force 64-bit
    codesign_identity=None,
    entitlements_file=None,
)