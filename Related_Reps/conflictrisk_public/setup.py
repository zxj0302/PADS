#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('ConflictRisk/README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('ConflictRisk/requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name="ConflictRisk",
    version="0.1.0",
    author="Original Authors + Python port",
    author_email="example@example.com",
    description="Python implementation of conflict risk optimization in social networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/conflictrisk-public",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'conflictrisk-demo=demo:main',
        ],
    },
)
