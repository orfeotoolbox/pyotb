# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyotb",
    version="1.5.4",
    author="Nicolas Narçon",
    author_email="nicolas.narcon@gmail.com",
    description="Library to enable easy use of the Orfeo Tool Box (OTB) in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.orfeo-toolbox.org/nicolasnn/pyotb/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    keywords="remote sensing, otb, orfeotoolbox, orfeo toolbox",
)
#package_dir={"": "src"},
