from setuptools import find_packages, setup


def from_file(file_name: str):
    with open(file_name, "r") as f:
        return f.read().splitlines()


def long_description():
    text = open("README.md", encoding="utf-8").read()
    # SVG images are not readable on PyPI, so replace them  with PNG
    text = text.replace(".svg", ".png")
    return text


setup(
    name="ride",
    version="0.2.0",
    description="Training wheels, side rails, and helicopter parent for your Deep Learning projects using Pytorch Lightning",
    long_description=long_description(),
    long_description_content_type="text/markdown",
    author="Lukas Hedegaard",
    author_email="lukasxhedegaard@gmail.com",
    url="https://github.com/LukasHedegaard/ride",
    install_requires=from_file("requirements.txt"),
    extras_require={
        "dev": from_file("requirements-dev.txt"),
        "build": ["setuptools", "wheel", "twine"],
    },
    packages=find_packages(exclude=["test"]),
    keywords=["deep learning", "pytorch", "AI"],
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
        # Pick your license as you wish
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
