#!/usr/bin/env python3
from hierarchical_file_finder import HierarchicalFileFinder
import os

def test_file_finder():
    finder = HierarchicalFileFinder(os.getcwd())
    
    # Test 1: Find existing file
    location, message = finder.find_file("c0d3r_cli.py")
    print(f"Test 1 - c0d3r_cli.py: {message}")
    print(f"Location: {location}")
    
    # Test 2: Find non-existent file
    location, message = finder.find_file("nonexistent.py")
    print(f"Test 2 - nonexistent.py: {message}")
    
    # Test 3: Test session cache (find same file again)
    location, message = finder.find_file("c0d3r_cli.py")
    print(f"Test 3 - c0d3r_cli.py (cached): {message}")
    
    print("\nSession cache contents:")
    for k, v in finder.session_cache.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    test_file_finder()
