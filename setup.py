#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for Bacteria Analyzer
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="bacteria-analyzer",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Comprehensive bacterial microscopy image analysis tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bacteria-analyzer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
    entry_points={
        "console_scripts": [
            "bacteria-analyzer=main:main",
        ],
    },
)