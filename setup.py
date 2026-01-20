from setuptools import setup, find_packages

setup(
    name="seq2ribo",
    version="0.1.0",
    description="Structure-aware integration of machine learning and simulation to predict ribosome location profiles from RNA sequences",
    author="gkaynar",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "einops>=0.8.0",
        "transformers>=4.40.0",
        # torch, mamba-ssm, ViennaRNA installed separately via conda
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
    },
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "seq2ribo=scripts.run_inference:main",
        ],
    },
)
