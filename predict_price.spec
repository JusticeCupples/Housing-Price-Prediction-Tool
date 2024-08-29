# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['predict_price.py'],
    pathex=['C:\\Users\\Justi\\projects\\python'],  # Update this path to your project directory
    binaries=[],
    datas=[('state_populations.db', '.'), ('housing_price_model.joblib', '.'), ('scaler.joblib', '.'), ('feature_names.joblib', '.')],
    hiddenimports=['sklearn', 'sklearn.ensemble', 'sklearn.preprocessing', 'numpy', 'matplotlib', 'sqlite3', 'tkinter'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='predict_price',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
