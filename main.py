"""
Top-level entry point for the CLAGN detection pipeline.

Usage:
    python main.py --input catalog.csv --output ./results/ --top_n 6 --test_n 5
    python main.py --input catalog.csv --output ./results/ --top_n 6 --mcmc
"""
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from clagn.main import main

if __name__ == '__main__':
    main()
