"""
Setup script for Quantum-Atmosphere-Transport package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8")

setup(
    name="quantum-atmosphere-transport",
    version="0.1.0",
    author="Fanghe Zhao",
    author_email="fzhao70@github",
    description="A quantum algorithm-based atmospheric tracer transport scheme",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fzhao70/Quantum-Atmosphere-Transport",
    project_urls={
        "Bug Tracker": "https://github.com/fzhao70/Quantum-Atmosphere-Transport/issues",
        "Documentation": "https://github.com/fzhao70/Quantum-Atmosphere-Transport",
        "Source Code": "https://github.com/fzhao70/Quantum-Atmosphere-Transport",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Physics",
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
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
        ],
    },
    keywords=[
        "quantum computing",
        "atmospheric science",
        "tracer transport",
        "numerical modeling",
        "quantum algorithms",
        "advection-diffusion",
    ],
)
