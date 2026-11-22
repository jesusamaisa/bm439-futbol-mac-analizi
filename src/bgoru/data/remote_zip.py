"""HTTP aralık (Range) istekleriyle uzak ZIP arşivinden seçici dosya okuma.

SoccerNet-Tracking-2023 arşivleri ~9 GB'tır; yalnızca gereken kareleri
(ör. her 15. kare) ve etiket dosyalarını indirmek için ZIP'in merkezi dizini
(central directory) okunur, ardından sadece istenen üyelerin bayt aralıkları
çekilir.
"""
from __future__ import annotations

import io
import time
import zipfile

import requests


class HttpRangeFile(io.RawIOBase):
    """zipfile.ZipFile'a verilebilen, salt-okunur, aralık-istekli dosya nesnesi."""

    def __init__(self, url: str, block: int = 1 << 20, session: requests.Session | None = None):
        super().__init__()
        self.url = url
        self.s = session or requests.Session()
        r = self.s.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.final_url = r.url
        self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.block = block
        self._cache_start = -1
        self._cache = b""

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def _get(self, start: int, end: int) -> bytes:
        for attempt in range(6):
            try:
                r = self.s.get(self.final_url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
                if r.status_code in (200, 206):
                    return r.content
            except requests.RequestException:
                pass
            time.sleep(2 * (attempt + 1))
        raise IOError(f"Range isteği başarısız: {start}-{end}")

    def readinto(self, b):
        n = len(b)
        if self.pos >= self.size or n == 0:
            return 0
        end = min(self.pos + n, self.size)
        cs = self._cache_start
        if cs >= 0 and self.pos >= cs and end <= cs + len(self._cache):
            data = self._cache[self.pos - cs:end - cs]
        else:
            fetch_end = min(max(end, self.pos + self.block), self.size) - 1
            chunk = self._get(self.pos, fetch_end)
            self._cache_start, self._cache = self.pos, chunk
            data = chunk[: end - self.pos]
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def open_remote_zip(url: str, block: int = 1 << 20) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url, block=block), buffer_size=block))


def fetch_member(url: str, info: zipfile.ZipInfo, session: requests.Session | None = None) -> bytes:
    """Tek bir ZIP üyesini doğrudan bayt aralığıyla indirip açar (thread-safe)."""
    s = session or requests.Session()
    # local header (30 bayt + dosya adı + extra) + sıkıştırılmış veri
    start = info.header_offset
    end = start + 30 + len(info.filename.encode()) + 1024 + info.compress_size
    for attempt in range(6):
        try:
            r = s.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
            if r.status_code in (200, 206):
                raw = r.content
                break
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    else:
        raise IOError(f"İndirilemedi: {info.filename}")
    import struct, zlib
    fn_len, extra_len = struct.unpack("<HH", raw[26:30])
    data = raw[30 + fn_len + extra_len: 30 + fn_len + extra_len + info.compress_size]
    if info.compress_type == zipfile.ZIP_STORED:
        return data
    if info.compress_type == zipfile.ZIP_DEFLATED:
        return zlib.decompress(data, -15)
    raise ValueError(f"Desteklenmeyen sıkıştırma: {info.compress_type}")
