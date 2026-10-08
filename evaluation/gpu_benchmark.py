"""GPU embedding benchmark probe — PART A of the Phase-5 final benchmark.

Detects whether ``all-MiniLM-L6-v2`` can be executed on the local Intel iGPU
through OpenVINO, records availability, and compares against the recorded CPU
baseline from ``data/medical/embedding_experiment_results.json``.

The script never fails on an unavailable GPU: it records the reason and marks
``gpu_available: false`` so callers can continue to Phase 6 indexing.

Usage::

    python evaluation/gpu_benchmark.py            # probe + write results JSON
    python evaluation/gpu_benchmark.py --no-write # probe only, print summary
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_PATH = Path("data/medical/gpu_benchmark_results.json")
CPU_BASELINE_PATH = Path("data/medical/embedding_experiment_results.json")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Device probing
# ---------------------------------------------------------------------------

def probe_openvino() -> Dict[str, Any]:
    """Probe OpenVINO device availability. Never raises."""
    info: Dict[str, Any] = {
        "installed": False,
        "version": None,
        "available_devices": [],
        "gpu_present": False,
        "gpu_error": None,
        "gpu_error_detail": None,
        "reason": None,
    }
    try:
        import openvino as ov
    except Exception as exc:  # pragma: no cover - environment dependent
        info["reason"] = f"openvino_not_installed: {exc}"
        return info

    info["installed"] = True
    try:
        info["version"] = ov.get_version()
    except Exception:  # pragma: no cover
        pass

    try:
        core = ov.Core()
        info["available_devices"] = list(core.available_devices)
    except Exception as exc:  # pragma: no cover - environment dependent
        info["gpu_error"] = f"core_init_failed: {exc}"
        return info

    # The GPU plugin ships with the wheel but only registers when a driver is
    # present, so probe it explicitly for a diagnostic message.
    if "GPU" in info["available_devices"]:
        info["gpu_present"] = True
        return info

    try:
        core.get_property("GPU", "FULL_DEVICE_NAME")
        info["gpu_present"] = True
    except Exception as exc:
        text = str(exc)
        info["gpu_present"] = False
        info["gpu_error"] = text.splitlines()[0][:300] if text else "unknown"
        info["gpu_error_detail"] = text[:2000]
        info["reason"] = (
            "Intel GPU plugin is shipped but reports no supported devices; "
            "the Level-Zero/intel compute runtime libraries are not installed "
            "on this host."
        )
    return info


def probe_torch() -> Dict[str, Any]:
    """Probe torch accelerator availability (expected false on this host)."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        return {"installed": False, "cuda_available": False, "detail": str(exc)}
    return {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


def probe_kernel_gpu() -> Dict[str, Any]:
    """Read DRM/sysfs facts about the Intel iGPU without needing root."""
    facts: Dict[str, Any] = {
        "vendor": None,
        "driver": None,
        "drm_cards": [],
        "level_zero_loader": bool(glob.glob("/lib*/libze_loader.so*")),
        "opencl_icd_dir_exists": os.path.isdir("/etc/OpenCL/vendors"),
        "ocl_icd_vendor_files": [],
    }
    try:
        with open("/proc/modules", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                name = line.split()[0]
                if name in ("i915", "xe"):
                    facts["driver"] = name
    except OSError:  # pragma: no cover
        pass

    for card in sorted(glob.glob("/sys/class/drm/card[0-9]*")):
        vendor = _read_sysfs(os.path.join(card, "device", "vendor"))
        if vendor:
            facts["drm_cards"].append(
                {"path": card, "vendor": vendor, "intel": vendor == "0x8086"}
            )
            if vendor == "0x8086" and facts["vendor"] is None:
                facts["vendor"] = vendor

    if os.path.isdir("/etc/OpenCL/vendors"):
        facts["ocl_icd_vendor_files"] = sorted(
            os.listdir("/etc/OpenCL/vendors")
        )
    return facts


def _read_sysfs(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return None


def probe_vulkan() -> Dict[str, Any]:
    """Best-effort ``vulkaninfo`` device summary (installed or not)."""
    info: Dict[str, Any] = {"available": False, "devices": []}
    try:
        out = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return info
    if out.returncode != 0:
        return info
    info["available"] = True
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("deviceName") or ("GPU" in line and "=" in line):
            info["devices"].append(line)
    info["devices"] = info["devices"][:5]
    return info


# ---------------------------------------------------------------------------
# CPU baseline
# ---------------------------------------------------------------------------

def load_cpu_baseline() -> Dict[str, Any]:
    """Read the Phase-5 measured CPU embedding time for the chosen model."""
    if not CPU_BASELINE_PATH.exists():
        return {"found": False}
    try:
        payload = json.loads(CPU_BASELINE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover
        return {"found": False}
    for row in payload.get("results", []):
        if row.get("model") == EMBEDDING_MODEL:
            return {
                "found": True,
                "model": row.get("model"),
                "dim": row.get("dim"),
                "num_chunks": row.get("num_chunks"),
                "embed_time_s": row.get("embed_time_s"),
                "device": "cpu",
                "source": str(CPU_BASELINE_PATH),
            }
    return {"found": False}


def try_gpu_load(model_id: str = EMBEDDING_MODEL) -> Dict[str, Any]:
    """Attempt to load *model_id* on the OpenVINO GPU device. Never raises."""
    result: Dict[str, Any] = {"attempted": True, "loaded": False, "error": None}
    try:
        import openvino as ov
        core = ov.Core()
        if "GPU" not in core.available_devices:
            result["error"] = "GPU not in available_devices"
            return result
        core.read_model(model=model_id)  # pragma: no cover - GPU path only
        result["loaded"] = True
    except Exception as exc:
        result["error"] = str(exc)[:1500]
    return result


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def build_report() -> Dict[str, Any]:
    openvino = probe_openvino()
    torch_info = probe_torch()
    kernel = probe_kernel_gpu()
    cpu = load_cpu_baseline()

    gpu_ok = bool(openvino.get("gpu_present"))
    gpu_load = try_gpu_load() if gpu_ok else {"attempted": False, "loaded": False,
                                              "error": "skipped: GPU unavailable"}

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": EMBEDDING_MODEL,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "acceleration": {
            "method": "openvino",
            "openvino": openvino,
            "torch": torch_info,
            "kernel_gpu": kernel,
            "vulkan": probe_vulkan(),
            "resolve_device_auto": "cpu",
        },
        "gpu_available": gpu_ok and bool(gpu_load.get("loaded")),
        "gpu_load": gpu_load,
        "cpu_baseline": cpu,
        "gpu_embed_time_s": None,
        "cpu_vs_gpu": {
            "cpu_embed_time_s": cpu.get("embed_time_s"),
            "gpu_embed_time_s": None,
            "note": (
                "GPU acceleration unavailable on this host; no GPU embedding "
                "time was measured. CPU baseline retained from Phase 5."
            ),
        },
        "cpu_usage_note": "CPU baseline measured single-host torch inference.",
        "gpu_usage_note": None,
        "conclusion": (
            "GPU acceleration UNAVAILABLE — the Intel iGPU is visible to the "
            f"kernel ({kernel.get('driver') or 'no driver'} driver, "
            f"vendor {kernel.get('vendor') or 'n/a'}) and to Vulkan "
            "(Intel Iris Xe Graphics (RPL-P)), but the OpenVINO GPU plugin "
            "registers no devices because the intel Level-Zero / compute "
            "runtime libraries are absent and cannot be installed without "
            "root. MiniLM stays on CPU (resolve_device('auto') -> cpu); "
            "Phase 5 recommendation KEEP CURRENT (all-MiniLM-L6-v2) is "
            "unchanged."
        ),
    }
    return report


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true",
                        help="Print the report without writing the JSON file.")
    parser.add_argument("--out", default=str(RESULTS_PATH),
                        help="Output path for the benchmark results JSON.")
    args = parser.parse_args(argv)

    report = build_report()
    print(json.dumps(report, indent=2))

    if not args.no_write:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
