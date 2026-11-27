from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        README = f.read()
except IOError:
    README = ''

setup(
    name='lib_mani',
    version='0.1.0',
    description='ManiSkill data processing library for WMM',
    long_description=README,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        'mani_skill',
        'numpy',
        'trimesh',
        'h5py',
        'transforms3d',
    ],
    include_package_data=True,
)
