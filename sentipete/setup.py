import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sentipete-pkg-MaxImune",
    version="1.0.1",
    author="MaxImune",
    author_email="maximune@gmail.com",
    description="A package for german sentiment analysis of topic-modelled keywords",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maximune/sentipete",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'spacy-nightly',
        'numpy',
        'pandas',
        'germalemma',
        'nltk',
        'sklearn',
        'matplotlib',
        'tqdm',
        'seaborn',
    ],
    include_package_data=True,
)
