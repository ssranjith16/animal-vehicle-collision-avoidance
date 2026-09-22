"""
Setup script for Animal Collision Avoidance System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
requirements_path = Path(__file__).parent / 'requirements.txt'
with open(requirements_path) as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read README
readme_path = Path(__file__).parent / 'README.md'
long_description = ''
if readme_path.exists():
    with open(readme_path) as f:
        long_description = f.read()

setup(
    name='animal-collision-avoidance',
    version='1.0.0',
    description='Automatic animal detection system for preventing vehicle-animal collisions on highways',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/animal-collision-avoidance',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'animal-detection=main:main',
            'train-model=models.train:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
    include_package_data=True,
    package_data={
        '': ['sounds/*.wav', 'config.yaml'],
    },
)