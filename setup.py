from setuptools import setup, find_packages

setup(
    name='magpyneto2',
    version='0.1.0',
    author='Daniel Pollak',
    author_email='dpollak@caltech.edu',
    description='magpyneto — magnet_search repo copy, with bug fixes',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    # Recorded here for the first time (previously undeclared). Versions are floors
    # matching what's validated against the `magneto2` conda env; PyPI-installable.
    install_requires=[
        "numpy>=2.2",
        "pandas>=2.3",
        "scipy>=1.16",
        "matplotlib>=3.10",
        "seaborn>=0.13",
        "scikit-learn>=1.7",
        "opencv-python>=4.12",
        "tifffile>=2025.9",
        "tqdm>=4.67",
        "PyYAML>=6.0",
        "python-dateutil>=2.9",
        "spikeinterface>=0.90",
        # NWB replatform (see .claude/CLAUDE.md and pipeline/nwb_io.py)
        "pynwb>=3.1",
        "hdmf>=4.1",
        # Lab-internal packages, not on public PyPI — install manually from their
        # source (ephysio/ecdfbounds are lab wheels; peakx is an editable local
        # checkout, e.g. `pip install -e C:\Users\dan\git\peakx`).
        "ephysio>=1.0.11",
        "ecdfbounds>=0.1.0",
        "peakx",
    ],
)
