from pathlib import Path
from setuptools import setup, find_packages

version = (Path("src/netspresso_trainer") / "VERSION").read_text().strip()

readme_contents = Path("README.md").read_text()

requirements = Path("requirements.txt").read_text().split('\n')

setup(
    name="netspresso_trainer",
    version=version,
    author="NetsPresso @Nota AI",
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
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)