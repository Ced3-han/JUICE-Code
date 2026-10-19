"""Write training samples into WebDataset tar shards (the member layout of the research packs).

Each sample is a dict {member suffix: bytes}, for example {"png": ..., "input.png": ...,
"txt": ..., "json": ...}; the tar member is "<key>.<suffix>". A shard is written to
".tmp_<name>" first and renamed when it is full, so a finished name is a finished shard.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path


class ShardWriter:
    def __init__(self, out_dir: Path, prefix: str, samples_per_shard: int):
        self.out_dir, self.prefix, self.per_shard = Path(out_dir), prefix, samples_per_shard
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.paths: list[Path] = []
        self._tar = self._tmp = None
        self._count = 0

    def write(self, key: str, members: dict[str, bytes]) -> None:
        if self._tar is None:
            final = self.out_dir / f"{self.prefix}_{len(self.paths):05d}.tar"
            self._tmp = final.with_name(f".tmp_{final.name}")
            self._tar = tarfile.open(self._tmp, "w")
            self.paths.append(final)
        for suffix, data in members.items():
            info = tarfile.TarInfo(f"{key}.{suffix}")
            info.size = len(data)
            self._tar.addfile(info, io.BytesIO(data))
        self._count += 1
        if self._count == self.per_shard:
            self._finish()

    def _finish(self) -> None:
        self._tar.close()
        self._tmp.rename(self.paths[-1])
        print(f"wrote {self.paths[-1]} ({self._count} samples)", flush=True)
        self._tar, self._count = None, 0

    def close(self) -> list[Path]:
        if self._tar is not None:
            self._finish()
        return self.paths


def png_bytes(image) -> bytes:
    """PIL image -> PNG bytes (the encoder settings of the research packs)."""
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=False)
    return buf.getvalue()
