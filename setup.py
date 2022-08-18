import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="minimal_mlops",
    version="0.0.1",
    author="Ayush Thakur",
    author_email="mein2work@gmail.com",
    description=(
        "Image classification in TensorFlow/Keras baked with Weights and Biases."
    ),
    license="MIT License",
    keywords="image_classification tensorflow keras wandb",
    packages=["classifier", "configs"],
    long_description=read("README.md"),
)