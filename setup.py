#!/usr/bin/env python3
"""
Setup script for Dielectric MCP Server
"""

from setuptools import setup, find_packages

setup(
    name="dielectric",
    version="1.0.0",
    description="AI-Powered PCB Component Placement System",
    author="Abiral Shakya",
    packages=find_packages(include=["backend*"]),
    include_package_data=True,
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "dedalus-labs>=0.0.1",
        "numpy>=1.24.0",
        "shapely>=2.0.0",
        "scipy>=1.11.0",
    ],
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "ngp=src.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
    ],
)
