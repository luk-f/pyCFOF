import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Concentration Free Outlier Factor",
    version="0.3.0",
    author="Lucas Foulon",
    author_email="lucas.foulon@gmail.com",
    description="Calculate the Concentration Free Outlier Factor score, based on Angiulli's work",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/luk-f/pyCFOF",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
