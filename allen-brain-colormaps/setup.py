from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mtg-colormaps",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Matplotlib and seaborn compatible colormaps for Middle Temporal Gyrus (MTG) cell types",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mtg-colormaps",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
    keywords="neuroscience, single-cell, visualization, colormap, matplotlib, seaborn, MTG",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/mtg-colormaps/issues",
        "Source": "https://github.com/yourusername/mtg-colormaps",
        "Documentation": "https://mtg-colormaps.readthedocs.io/",
    },
)