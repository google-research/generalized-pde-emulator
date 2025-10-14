# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages

setup(
    name="pde_emulator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jax>=0.4.13",
        "jaxlib>=0.4.13",
        "equinox>=0.11.0",
        "optax>=0.1.5",
        "apebench",
        "exponax",
        "numpy>=1.22.0",
        "matplotlib>=3.5.0",
        "scipy>=1.8.0",
        "tqdm>=4.62.0",
        "pandas>=1.4.0",
    ],
    author="Generalized PDE Emulator Team",
    author_email="generalized-pde-emulator-team@google.com",
    description="A library for generalized PDE emulation using neural networks",
    keywords="pde, neural networks, machine learning, scientific computing",
    url="https://github.com/google-research/generalized-pde-emulator",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",
)
