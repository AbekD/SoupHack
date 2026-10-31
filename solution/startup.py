"""Start-up costs of the measured `soup train` that our code can cut without changing what it
does. Found with a cProfile of the whole run on the Kaggle T4 (SOUP_KERNELS_CPROFILE=1).

1. `import trl` runs `importlib.metadata.packages_distributions()` (trl/import_utils.py), which
   walks every installed distribution and builds a pathlib object for every line of every
   RECORD file. On Kaggle's image (hundreds of packages) that was ~6 s under the profiler. We
   compute the very same mapping with plain string operations (same files, same rules as the
   standard library; anything unusual goes through the standard library's own code).

2. `datasets` fingerprints every `.map()` by dill-pickling the mapped function, which here
   captures the tokenizer: one pure-Python pickle of a 151k-entry vocabulary and its merges
   (~8 s under the profiler). A fingerprint only names `datasets`' cache files; Soup's own
   records (its dataset hash, rewind log, metrics) do not use it. We fingerprint a tokenizer by
   the SHA-256 of its full serialized state (the fast tokenizer's JSON, made in Rust, plus its
   Python-side settings), so a different tokenizer still gives a different fingerprint.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import importlib.machinery
import importlib.metadata as metadata
import os


def _module_name(filename):
    """inspect.getmodulename on a basename."""
    for suffix in sorted(importlib.machinery.all_suffixes(), key=len, reverse=True):
        if filename.endswith(suffix):
            return filename[:-len(suffix)]
    return None


_SIMPLE = None


def _toplevel(name):
    """importlib.metadata._get_toplevel_name(PackagePath(name)) for a RECORD path."""
    if not name or name.startswith("/") or "//" in name or name.endswith("/") \
            or name.startswith("./") or "/./" in name or name == "." or "\\" in name:
        from importlib.metadata import PackagePath, _get_toplevel_name
        return _get_toplevel_name(PackagePath(name))
    if "/" in name:
        return name.split("/", 1)[0]
    return _module_name(os.path.basename(name)) or name


def _inferred(dist):
    text = dist.read_text("RECORD")
    if not text:   # egg-info layouts and the like: the standard library's own code
        return metadata._top_level_inferred(dist)
    names = {_toplevel(row[0]) for row in csv.reader(text.splitlines()) if row}
    return [name for name in names if "." not in name]


def packages_distributions():
    """Same result as importlib.metadata.packages_distributions(), faster."""
    pkg_to_dist = collections.defaultdict(list)
    for dist in metadata.distributions():
        for pkg in metadata._top_level_declared(dist) or _inferred(dist):
            pkg_to_dist[pkg].append(dist.metadata["Name"])
    return dict(pkg_to_dist)


def patch_packages_distributions():
    if not hasattr(metadata, "_top_level_declared") or \
            not hasattr(metadata, "_top_level_inferred"):
        return  # another Python: leave the standard implementation
    original = metadata.packages_distributions
    metadata.packages_distributions = packages_distributions
    metadata._soup_original_packages_distributions = original


def _tokenizer_digest(tokenizer):
    digest = hashlib.sha256(type(tokenizer).__module__.encode())
    digest.update(type(tokenizer).__qualname__.encode())
    backend = getattr(tokenizer, "_tokenizer", None)
    to_str = getattr(backend, "to_str", None)
    if callable(to_str):
        digest.update(to_str().encode("utf-8"))
    for key in ("init_kwargs", "chat_template", "name_or_path", "padding_side",
                "truncation_side", "model_max_length", "added_tokens_decoder"):
        value = getattr(tokenizer, key, None)
        digest.update(key.encode())
        digest.update(repr(sorted(value.items(), key=lambda kv: repr(kv[0]))
                           if isinstance(value, dict) else value).encode("utf-8", "replace"))
    return digest.hexdigest()


def patch_datasets_tokenizer_fingerprint():
    try:
        from datasets.utils import _dill
    except Exception:  # noqa: BLE001
        return
    original = getattr(_dill, "_save_transformersPreTrainedTokenizerBase", None)
    if original is None:
        return

    def _save_tokenizer(pickler, obj):
        try:
            digest = _tokenizer_digest(obj)
        except Exception:  # noqa: BLE001
            return original(pickler, obj)
        pickler.save_reduce(str, ("transformers-tokenizer-sha256:" + digest,), obj=obj)

    _dill._save_transformersPreTrainedTokenizerBase = _save_tokenizer
    # types already registered with the original reducer (normally none at this point)
    dispatch = getattr(getattr(_dill, "Pickler", None), "dispatch", None)
    if dispatch is not None:
        for key, value in list(dispatch.items()):
            if value is original:
                dispatch[key] = _save_tokenizer
