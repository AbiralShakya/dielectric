#!/usr/bin/env python3
"""
Setup script for Neuro-Geometric Placer
"""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="neuro-geometric-placer",
    version="1.0.0",
    description="AI-Powered PCB Component Placement System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Abiral Shakya",
    packages=find_packages(where=".", include=["backend*"]),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "ngp-server=backend.mcp_servers.ngp_server:main",
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
