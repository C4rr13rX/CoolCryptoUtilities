"""Read-only WAL forensic probe. Opens the -wal file for reading only and
parses headers per the SQLite WAL format spec. Never opens the database, never
issues a pragma, cannot block or disturb a live writer.

WAL header (32 bytes):
   0-3  magic 0x377f0682 (LE checksums) / 0x377f0683 (BE checksums)
   4-7  file format version
   8-11 database page size
  12-15 checkpoint sequence number
  16-19 salt-1   (incremented on every WAL restart)
  20-23 salt-2   (randomised on every WAL restart)
  24-31 header checksum

Frame header (24 bytes), then one page of data:
   0-3  page number
   4-7  db size in pages after commit (0 = not a commit frame)
   8-15 salt-1, salt-2 copied from the WAL header AT WRITE TIME
  16-23 frame checksum

A frame whose salts match the CURRENT header salts belongs to the live WAL.
A frame with stale salts is dead space left behind by a previous checkpoint
+ restart, i.e. pure high-water mark.
"""
import os
import struct
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "storage/trading_cache.db-wal"
size = os.path.getsize(path)

with open(path, "rb") as fh:
    hdr = fh.read(32)
    magic, fmt, page_size, ckpt_seq, salt1, salt2, ck1, ck2 = struct.unpack(">IIIIIIII", hdr)

    print(f"file                      : {path}")
    print(f"size on disk              : {size:,} bytes ({size / 1024 / 1024:.1f} MB)")
    print(f"magic                     : 0x{magic:08x}", "OK" if magic in (0x377F0682, 0x377F0683) else "BAD -- not a WAL")
    print(f"format version            : {fmt}")
    print(f"page size                 : {page_size:,}")
    print(f"checkpoint sequence number: {ckpt_seq:,}   <- times this WAL has been checkpointed AND restarted")
    print(f"current salt-1 / salt-2   : {salt1} / {salt2}")

    frame_size = 24 + page_size
    capacity = (size - 32) // frame_size
    print(f"frame size                : {frame_size:,} bytes")
    print(f"capacity (frames on disk) : {capacity:,}")
    print()

    live = 0
    stale = 0
    last_commit_at = -1
    db_pages_at_commit = 0
    first_stale_at = -1

    for i in range(capacity):
        fh.seek(32 + i * frame_size)
        fhdr = fh.read(24)
        if len(fhdr) < 24:
            break
        pgno, ntrunc, fs1, fs2, _, _ = struct.unpack(">IIIIII", fhdr)
        if fs1 == salt1 and fs2 == salt2:
            live += 1
            if ntrunc:
                last_commit_at = i
                db_pages_at_commit = ntrunc
        else:
            stale += 1
            if first_stale_at < 0:
                first_stale_at = i

    live_bytes = live * frame_size
    print(f"LIVE frames  (current salt): {live:,}   = {live_bytes / 1024 / 1024:.1f} MB")
    print(f"STALE frames (old salt)    : {stale:,}   = {stale * frame_size / 1024 / 1024:.1f} MB  <- dead high-water mark")
    print(f"first stale frame at index : {first_stale_at:,}")
    print(f"last COMMIT frame at index : {last_commit_at:,}  (db size then: {db_pages_at_commit:,} pages)")
    print()
    if capacity:
        print(f"LIVE FRACTION OF THE FILE  : {100.0 * live / capacity:.4f}%")
    print(f"autocheckpoint threshold   : 1000 pages (per the item) = {1000 * frame_size / 1024 / 1024:.1f} MB")
    if live:
        print(f"live WAL vs that threshold : {live / 1000.0:.2f}x")
    print(f"DISK FILE vs that threshold: {capacity / 1000.0:.2f}x")
