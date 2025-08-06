from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
def read_requirements():
    with open('requirements.txt') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read README for long description
def read_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

setup(
    name="alpaca-strategy",
    version="1.0.0",
    description="Machine learning trading system using transformer architecture for automated stock trading",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Alpaca Strategy Team",
    author_email="contact@example.com",
    url="https://github.com/your-org/alpaca-strategy",
    packages=find_packages(),
    include_package_data=True,
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'black',
            'flake8',
            'mypy',
            'pytest',
            'pytest-cov',
        ],
        'gpu': [
            'torch>=2.0.0',
            'torchvision',
        ],
    },
    entry_points={
        'console_scripts': [
            'alpaca-train=scripts.train:main',
            'alpaca-trade=scripts.trade_realtime_ws:main',
            'alpaca-backtest=scripts.backtest_with_model:main',
            'alpaca-fetch-data=scripts.fetch_trade_data:main',
            'alpaca-update-model=scripts.update_model_one_day:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    keywords="trading, machine learning, transformer, alpaca, finance, quantitative",
    project_urls={
        "Bug Reports": "https://github.com/your-org/alpaca-strategy/issues",
        "Source": "https://github.com/your-org/alpaca-strategy",
        "Documentation": "https://github.com/your-org/alpaca-strategy#readme",
    },
) 