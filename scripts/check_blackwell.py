#!/usr/bin/env python3
"""Smoke-check the supported ReconViaGen Blackwell runtime."""

from __future__ import annotations

import importlib
import platform
import sys


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    require(platform.system() == "Linux", "Ubuntu Linux is required")
    require(platform.machine() == "x86_64", "Linux x86_64 is required")
    require(sys.version_info[:2] == (3, 10), f"Python 3.10 is required, found {platform.python_version()}")

    import torch

    require(torch.__version__.split("+")[0] == "2.7.1", f"PyTorch 2.7.1 is required, found {torch.__version__}")
    require(torch.version.cuda == "12.8", f"PyTorch CUDA 12.8 is required, found {torch.version.cuda}")
    require(torch.cuda.is_available(), "CUDA is not available to PyTorch")

    device_count = torch.cuda.device_count()
    require(device_count > 0, "No CUDA devices were detected")
    for device_index in range(device_count):
        capability = torch.cuda.get_device_capability(device_index)
        require(capability == (12, 0), f"GPU {device_index} has unsupported capability {capability}")
        print(f"GPU {device_index}: {torch.cuda.get_device_name(device_index)} (sm_{capability[0]}{capability[1]})")

    modules = (
        "flash_attn",
        "nvdiffrast.torch",
        "nvdiffrec_render",
        "cumesh",
        "flex_gemm",
        "o_voxel",
        "spconv.pytorch",
    )
    for module_name in modules:
        importlib.import_module(module_name)
        print(f"import {module_name}: OK")

    qkv = torch.randn(1, 32, 3, 4, 64, device="cuda", dtype=torch.float16)
    from flash_attn import flash_attn_qkvpacked_func

    output = flash_attn_qkvpacked_func(qkv)
    require(output.shape == (1, 32, 4, 64), f"Unexpected FlashAttention output shape: {output.shape}")
    require(torch.isfinite(output).all().item(), "FlashAttention returned non-finite values")
    print("FlashAttention sm_120 kernel: OK")
    print("ReconViaGen Blackwell environment: OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Blackwell environment check failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
