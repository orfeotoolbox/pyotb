# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyotb",
    version="1.1.1",
    author="Nicolas Narçon",
    author_email="nicolas.narcon@gmail.com",
    description="Library to enable easy use of the Orfeo Tool Box (OTB) in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.orfeo-toolbox.org/nicolasnn/pyotb/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires="==3.5",
    keywords="remote sensing, otb, orfeotoolbox, orfeo toolbox",
)
#package_dir={"": "src"},
