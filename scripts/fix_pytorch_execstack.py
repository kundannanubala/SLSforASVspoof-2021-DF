#!/usr/bin/env python3
"""Patch libtorch_cpu.so to clear the executable-stack flag.

Older PyTorch wheels (e.g. 1.12.x) fail to import on newer Linux kernels with:
  ImportError: libtorch_cpu.so: cannot enable executable stack ...

Run inside your conda env (no sudo required):
  python scripts/fix_pytorch_execstack.py
"""

import glob
import os
import struct
import sys


def find_libtorch_cpu():
    import torch

    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_cpu.so")
    if os.path.isfile(torch_lib):
        return torch_lib

    for pattern in (
        os.path.join(sys.prefix, "lib", "python*", "site-packages", "torch", "lib", "libtorch_cpu.so"),
        os.path.join(sys.prefix, "lib", "libtorch_cpu.so"),
    ):
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


def patch_with_patchelf(path):
    import subprocess

    result = subprocess.run(
        ["patchelf", "--clear-execstack", path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "patchelf failed")
    return "patchelf"


def patch_with_pyelftools(path):
    from elftools.elf.elffile import ELFFile
    from elftools.elf.constants import PT_GNU_STACK

    with open(path, "rb") as f:
        elf = ELFFile(f)
        for i, seg in enumerate(elf.iter_segments()):
            if seg.header["p_type"] != PT_GNU_STACK:
                continue
            if not (seg.header["p_flags"] & 0x1):
                return "already_patched"

            new_flags = seg.header["p_flags"] & ~0x1
            ph_offset = elf.header["e_phoff"]
            ph_size = elf.header["e_phentsize"]
            field_offset = 4 if elf.elfclass == 64 else 24
            final_offset = ph_offset + (i * ph_size) + field_offset

            f.seek(final_offset)
            fmt = "<I" if elf.little_endian else ">I"
            f.write(struct.pack(fmt, new_flags))
            return "pyelftools"

    raise RuntimeError("PT_GNU_STACK segment not found")


def patch_with_struct(path):
    """Minimal ELF64 patch without external dependencies."""
    PT_GNU_STACK = 0x6474E551
    PF_X = 0x1

    with open(path, "rb+") as f:
        ident = f.read(16)
        if ident[:4] != b"\x7fELF":
            raise RuntimeError("Not an ELF file")
        elf_class = ident[4]
        if elf_class != 2:
            raise RuntimeError("Only ELF64 is supported by the built-in patcher")

        f.seek(32)
        e_phoff = struct.unpack("<Q", f.read(8))[0]
        f.seek(54)
        e_phentsize, e_phnum = struct.unpack("<HH", f.read(4))

        for i in range(e_phnum):
            ph_offset = e_phoff + i * e_phentsize
            f.seek(ph_offset)
            p_type, p_flags = struct.unpack("<II", f.read(8))
            if p_type != PT_GNU_STACK:
                continue
            if not (p_flags & PF_X):
                return "already_patched"
            f.seek(ph_offset + 4)
            f.write(struct.pack("<I", p_flags & ~PF_X))
            return "struct"

    raise RuntimeError("PT_GNU_STACK segment not found")


def main():
    # Import torch only to locate the library path; if import fails, search manually.
    path = None
    try:
        path = find_libtorch_cpu()
    except ImportError:
        for candidate in glob.glob(
            os.path.join(sys.prefix, "lib", "python*", "site-packages", "torch", "lib", "libtorch_cpu.so")
        ):
            path = candidate
            break

    if not path:
        print("Could not locate libtorch_cpu.so in the active environment.", file=sys.stderr)
        sys.exit(1)

    print(f"Patching: {path}")

    for method_name, patch_fn in (
        ("patchelf", patch_with_patchelf),
        ("pyelftools", patch_with_pyelftools),
        ("struct", patch_with_struct),
    ):
        try:
            result = patch_fn(path)
            print(f"Done via {method_name} ({result}).")
            break
        except FileNotFoundError:
            continue
        except ImportError:
            continue
        except RuntimeError as exc:
            if str(exc) == "already_patched":
                print("Already patched — no changes needed.")
                break
            if method_name == "struct":
                print(f"Failed: {exc}", file=sys.stderr)
                sys.exit(1)
            continue
    else:
        print("All patch methods failed.", file=sys.stderr)
        sys.exit(1)

    try:
        import torch

        print(f"Verified: torch {torch.__version__} imports successfully.")
    except ImportError as exc:
        print(f"Patch applied but torch still fails to import: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
