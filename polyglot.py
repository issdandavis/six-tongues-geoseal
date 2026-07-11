"""polyglot.py - represent one small concept across real languages, and
(best-effort) EXECUTE each to verify they agree.

This is geoseal's portable representation front-end. It is self-contained and
stdlib-only: REPRESENTATION always works (the code is just text); EXECUTION runs
only for toolchains present on this machine, and honestly reports the rest as
"represented (no toolchain)".

The deeper self-verifying router - op-class routing rules re-checked by
execution across the tongues - lives in loom/polyglot.py. The idiomatic per-
language snippets below are vendored from that router (the Python/JS/Rust/
Haskell/Julia camps are its verified probes) and extended here with C++.

Why this matters: representing one idea in many languages and running each is a
cross-check. Where the languages AGREE, the concept is language-independent;
where they split into camps (e.g. negative integer division, rounding, bignum),
the split is a real semantic fact, not a bug - and the box has surfaced it.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# tools not always on PATH (this machine) - fall back to known locations
_STACK = shutil.which("stack") or r"C:\Users\issda\AppData\Roaming\local\bin\stack.exe"
_JULIA = shutil.which("julia") or r"C:\Users\issda\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe"


def _cxx():
    return shutil.which("g++") or shutil.which("clang++")


# name -> (file extension, compiled?)
LANGS = ["Python", "JavaScript", "Rust", "Haskell", "Cpp", "Julia"]
_EXT = {"Python": ".py", "JavaScript": ".js", "Rust": ".rs",
        "Haskell": ".hs", "Cpp": ".cpp", "Julia": ".jl"}

# one idiomatic snippet per (op-class, language). Each prints exactly one value.
OPS = {
    "scalar": {
        "Python": "print(2+3*4)",
        "JavaScript": "console.log(2+3*4)",
        "Rust": 'fn main(){println!("{}", 2+3*4);}',
        "Haskell": "main = print (2+3*4)",
        "Cpp": '#include <iostream>\nint main(){std::cout<<(2+3*4)<<"\\n";}',
        "Julia": "println(2+3*4)",
    },
    "bignum": {
        "Python": "print(2**200)",
        "JavaScript": "console.log(2n**200n)",
        "Rust": 'fn main(){println!("{}", 2f64.powi(200));}',
        "Haskell": "main = print (2^200 :: Integer)",
        "Cpp": '#include <iostream>\n#include <cmath>\nint main(){std::cout<<pow(2.0,200)<<"\\n";}',
        "Julia": "println(big(2)^200)",
    },
    "neg_intdiv": {
        "Python": "print(-7//2)",
        "JavaScript": "console.log(Math.trunc(-7/2))",
        "Rust": 'fn main(){let a:i32=-7;let b:i32=2;println!("{}", a/b);}',
        "Haskell": "main = print (div (-7) 2)",
        "Cpp": '#include <iostream>\nint main(){int a=-7,b=2;std::cout<<(a/b)<<"\\n";}',
        "Julia": "println(div(-7,2))",
    },
    "neg_mod": {
        "Python": "print(-7%3)",
        "JavaScript": "console.log(-7%3)",
        "Rust": 'fn main(){let a:i32=-7;let b:i32=3;println!("{}", a%b);}',
        "Haskell": "main = print (mod (-7) 3)",
        "Cpp": '#include <iostream>\nint main(){int a=-7,b=3;std::cout<<(a%b)<<"\\n";}',
        "Julia": "println(-7%3)",
    },
    "rounding": {
        "Python": "print(round(2.5))",
        "JavaScript": "console.log(Math.round(2.5))",
        "Rust": 'fn main(){println!("{}", (2.5f64).round() as i64);}',
        "Haskell": "main = print (round 2.5 :: Int)",
        "Cpp": '#include <iostream>\n#include <cmath>\nint main(){std::cout<<std::lround(2.5)<<"\\n";}',
        "Julia": "println(round(Int,2.5))",
    },
}


def represent(op):
    """Return {lang: source_code} for an op-class. Pure text; always works."""
    if op not in OPS:
        raise KeyError(op)
    return dict(OPS[op])


def available(lang):
    """Is this language's toolchain runnable on this machine?"""
    if lang == "Python":
        return True
    if lang == "JavaScript":
        return shutil.which("node") is not None
    if lang == "Rust":
        return shutil.which("rustc") is not None
    if lang == "Haskell":
        return os.path.exists(_STACK) or shutil.which("stack") is not None
    if lang == "Cpp":
        return _cxx() is not None
    if lang == "Julia":
        return os.path.exists(_JULIA) or shutil.which("julia") is not None
    return False


def _runner(lang, path, exe):
    if lang == "Python":
        return [sys.executable, "-I", path]
    if lang == "JavaScript":
        return ["node", path]
    if lang == "Rust":
        return [exe]
    if lang == "Haskell":
        return [_STACK, "runghc", "--", path]
    if lang == "Cpp":
        return [exe]
    if lang == "Julia":
        return [_JULIA, "--startup-file=no", path]
    return None


def _compiler(lang, path, exe):
    if lang == "Rust":
        return ["rustc", "-O", path, "-o", exe]
    if lang == "Cpp":
        cxx = _cxx()
        return [cxx, "-O2", "-std=c++17", path, "-o", exe] if cxx else None
    return None


def run_lang(lang, src, timeout=90):
    """Execute src in lang, return the last printed value (str) or a status.

    Returns 'toolchain-not-found' without running if the language is absent,
    so representation and execution stay honestly separate.
    """
    if not available(lang):
        return "toolchain-not-found"
    scratch = os.path.join(tempfile.gettempdir(), "geoseal-polyglot")
    os.makedirs(scratch, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=_EXT[lang], dir=scratch)
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    exe = path + ".exe"
    try:
        comp = _compiler(lang, path, exe)
        if comp:
            c = subprocess.run(comp, capture_output=True, text=True, timeout=timeout)
            if c.returncode != 0:
                return "COMPILE-ERR"
        r = subprocess.run(_runner(lang, path, exe), capture_output=True, text=True, timeout=timeout)
        out = _ANSI.sub("", (r.stdout or r.stderr)).strip().splitlines()
        val = out[-1].strip() if out else "(no output)"
        return val[:-2] if val.endswith(".0") else val
    except Exception as e:  # noqa: BLE001 - report any toolchain failure as data
        return "ERR:" + str(e)[:24]
    finally:
        for f in (path, exe):
            try:
                os.unlink(f)
            except OSError:
                pass


def camps(values):
    """{lang: value} -> list of (value, [langs]) camps, agreement first."""
    groups = {}
    for lang, val in values.items():
        groups.setdefault(val, []).append(lang)
    return sorted(groups.items(), key=lambda kv: -len(kv[1]))
