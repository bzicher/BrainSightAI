"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    author="Blanka Zicher",
    author_email='bz21@ic.ac.uk',
    python_requires='>=3.8',
    description="AI classifier that predicts brain states from peripheral physiological recordings",
    install_requires=requirements,
    license="MIT",
    long_description=readme,
    include_package_data=True,
    name='BrainSightAI',
    packages=find_packages(include=['brainsight','brainsight.*']),
    url='https://github.com/bzicher/BrainSightAI',
    version='0.1.0',
    zip_safe=False,
)
