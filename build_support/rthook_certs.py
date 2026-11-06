"""PyInstaller 运行时钩子：修复冻结环境的 CA 根证书。

打包后的 exe 里 OpenSSL 拿不到系统根证书，huggingface_hub / httpx 会抛
``CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate``，
表现就是模型下载、管线配置拉取全部失败。

这里做两件事：
1. 把环境变量指向随包分发的 ``certifi`` cacert.pem（requests / curl 生效）；
2. 给 ``SSLContext.load_default_certs`` 打补丁，额外加载一遍 certifi 根证书
   （httpx / urllib3 默认 context 也能拿到）。
"""

from __future__ import annotations

import os


def _install() -> None:
    try:
        import certifi
    except Exception:  # pragma: no cover - certifi 缺失时静默跳过
        return

    try:
        ca_file = certifi.where()
    except Exception:
        return
    if not ca_file or not os.path.isfile(ca_file):
        return

    for name in (
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "HTTPX_CA_BUNDLE",
    ):
        os.environ.setdefault(name, ca_file)

    try:
        import ssl

        original = ssl.SSLContext.load_default_certs

        def load_default_certs(self, purpose=ssl.Purpose.SERVER_AUTH):
            original(self, purpose)
            try:
                self.load_verify_locations(cafile=ca_file)
            except Exception:
                pass

        ssl.SSLContext.load_default_certs = load_default_certs
    except Exception:
        pass


_install()
