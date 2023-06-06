from setuptools import setup, find_packages

PROJECT_NAME = "tgplus"

setup(
    name=PROJECT_NAME,
    version='0.1.0',

    # License and description are updated from separate files:
    license="TBD",
    long_description="Exploratory code",

    # Packages to install:
    packages=[
        PROJECT_NAME
    ],

    # Extra files that should be copied as part of an install:
    data_files=[
    ]
)
