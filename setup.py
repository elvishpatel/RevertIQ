"""
RevertIQ — Cross-Sectional Mean Reversion Quant Research Platform
=================================================================

A modular, production-quality quantitative trading research system
for Indian equity markets (NSE / NIFTY 50).

Install in development mode:
    pip install -e .

Install in Google Colab:
    !pip install git+https://github.com/elvishpatel/RevertIQ.git
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="revertiq",
    version="0.1.0",
    author="RevertIQ Team",
    description="Cross-Sectional Mean Reversion Quant Research Platform for Indian Markets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/RevertIQ",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="quantitative-trading mean-reversion indian-stocks nse nifty backtesting",
)
