# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

hiddenimports = []
hiddenimports += collect_submodules('madne')

# 本地 SD 后端：diffusers 的 pipeline/scheduler、huggingface_hub 的传输层
# 都是运行期动态导入，必须显式收集，否则打包后只会降级到 Mock。
for package in ('diffusers', 'accelerate', 'huggingface_hub', 'safetensors'):
    hiddenimports += collect_submodules(package)

# transformers 只使用 CLIP 文本编码器/分词器（SD1.5 / SD2.1 / SDXL）
hiddenimports += collect_submodules('transformers.models.clip')
hiddenimports += ['tqdm.auto', 'safetensors.torch', 'requests']

datas = []
for package in ('diffusers', 'transformers', 'certifi'):
    datas += collect_data_files(package)

# diffusers 在 import 时用 importlib.metadata 校验 requests / filelock / numpy
# 的版本。缺 dist-info 会直接抛 PackageNotFoundError（表现为“需要安装 torch /
# diffusers / transformers / accelerate”），所以这些发行版元数据必须打进去。
for dist in (
    'requests',
    'filelock',
    'numpy',
    'packaging',
    'diffusers',
    'transformers',
    'huggingface-hub',
    'accelerate',
    'safetensors',
    'Pillow',
    'tokenizers',
    'regex',
    'tqdm',
):
    try:
        datas += copy_metadata(dist)
    except Exception:
        pass

# 单文件权重加载所需的管线配置（约 1.6MB）：让 exe + sd_turbo.safetensors
# 可以完全离线出图，不必再去 Hub 拉 model_index.json。
# 注意用绝对路径：--specpath build 会让相对路径解析到 build 目录下。
import os

_ROOT = os.getcwd()
datas += [(os.path.join(_ROOT, 'madne', 'image', 'configs'), 'madne/image/configs')]


a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    # 冻结环境 SSL 根证书修复
    runtime_hooks=[os.path.join(_ROOT, 'build_support', 'rthook_certs.py')],
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
    name='MADNE',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
