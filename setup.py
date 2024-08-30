# Copyright (C) 2024 Nota Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----------------------------------------------------------------------------

from pathlib import Path

from setuptools import find_packages, setup

version = (Path("src/netspresso_trainer") / "VERSION").read_text().strip()

readme_contents = Path("README.md").read_text()

requirements = Path("requirements.txt").read_text().split('\n')
requirements_all = requirements

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
    package_data={"": ["VERSION"]},
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "netspresso-train = netspresso_trainer.trainer_main:train_cli",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)