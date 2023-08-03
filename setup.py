from pathlib import Path
from setuptools import setup, find_packages

version = (Path("src/netspresso_trainer") / "VERSION").read_text().strip()

readme_contents = Path("README.md").read_text()

requirements = Path("requirements.txt").read_text().split('\n')
requirements_optional = Path("requirements-optional.txt").read_text().split('\n')
requirements_all = requirements + requirements_optional

setup(
    name="netspresso_trainer",
    version=version,
    author="NetsPresso",
    author_email="netspresso@nota.ai",
    description="NetsPresso Python Package",
    long_description=readme_contents,
    long_description_content_type="text/markdown",
    url="https://github.com/Nota-NetsPresso/netspresso-trainer",
    install_requires=requirements,
    package_dir={"": "src"},
    packages=find_packages("src", exclude=("tests",)),
    package_data={},
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "netspresso-train = netspresso_trainer.train:netspresso_train",
            "netspresso-train-fx = netspresso_trainer.train:netspresso_train_fx",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)