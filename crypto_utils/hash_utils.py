import hashlib

def sha256_hash(data: str) -> str:
    """Generate SHA256 hash of input data."""
    return hashlib.sha256(data.encode()).hexdigest()

def md5_hash(data: str) -> str:
    """Generate MD5 hash of input data."""
    return hashlib.md5(data.encode()).hexdigest()
