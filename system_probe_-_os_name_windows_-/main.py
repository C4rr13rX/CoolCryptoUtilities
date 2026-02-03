#!/usr/bin/env python3
"""
Cool Crypto Utilities - Main Entry Point
"""

import sys
import argparse
from crypto_utils import hash_utils, encryption_utils, key_utils

def main():
    parser = argparse.ArgumentParser(description='Cool Crypto Utilities')
    parser.add_argument('--hash', help='Hash a string using SHA256')
    parser.add_argument('--encrypt', help='Encrypt a string')
    parser.add_argument('--decrypt', help='Decrypt a string')
    parser.add_argument('--generate-key', action='store_true', help='Generate a new encryption key')
    
    args = parser.parse_args()
    
    if args.hash:
        result = hash_utils.sha256_hash(args.hash)
        print(f"SHA256: {result}")
    elif args.encrypt:
        key = key_utils.load_or_generate_key()
        result = encryption_utils.encrypt_string(args.encrypt, key)
        print(f"Encrypted: {result}")
    elif args.decrypt:
        key = key_utils.load_or_generate_key()
        result = encryption_utils.decrypt_string(args.decrypt, key)
        print(f"Decrypted: {result}")
    elif args.generate_key:
        key_utils.generate_new_key()
        print("New key generated and saved")
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
