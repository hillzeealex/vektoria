"""
IndexManager — owns a root data directory containing one sub-directory per index,
with an LRU cache of open Index objects so a single instance stays memory-bounded.
"""

import re
import shutil
from collections import OrderedDict
from pathlib import Path

from .index import Index

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class IndexManager:
    def __init__(self, data_dir, cache_size: int = 8):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._cache_size = cache_size
        self._open: "OrderedDict[str, Index]" = OrderedDict()

    @staticmethod
    def _validate_name(name: str) -> None:
        if not isinstance(name, str) or not _SAFE_NAME.match(name):
            raise ValueError(
                f"Invalid index name {name!r}: use letters, digits, '-' and '_' only"
            )

    def create_index(self, name: str, dimension: int, metric: str = "cosine",
                     backend: str = "bruteforce", bit_width: int = 4) -> None:
        self._validate_name(name)
        path = self.data_dir / name
        if path.exists():
            raise ValueError(f"Index {name!r} already exists")
        Index.create(path, dimension=dimension, metric=metric,
                     backend=backend, bit_width=bit_width).close()

    def list_indexes(self) -> list[dict]:
        # Read summaries directly — don't route through get(), which would load
        # every index's vectors and churn the LRU cache just to list them.
        out = []
        for path in sorted(self.data_dir.iterdir()):
            if not (path / "index.db").exists():
                continue
            out.append({"name": path.name, **Index.stat(path)})
        return out

    def get(self, name: str) -> Index:
        self._validate_name(name)
        if name in self._open:
            self._open.move_to_end(name)
            return self._open[name]
        path = self.data_dir / name
        if not (path / "index.db").exists():
            raise KeyError(f"No index named {name!r}")
        idx = Index(path)
        self._open[name] = idx
        self._evict_if_needed()
        return idx

    def _evict_if_needed(self) -> None:
        while len(self._open) > self._cache_size:
            _, victim = self._open.popitem(last=False)
            victim.close()

    def delete_index(self, name: str) -> None:
        self._validate_name(name)
        if name in self._open:
            self._open.pop(name).close()
        path = self.data_dir / name
        if not path.exists():
            raise KeyError(f"No index named {name!r}")
        shutil.rmtree(path)
