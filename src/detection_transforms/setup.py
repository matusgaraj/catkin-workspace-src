# ~/catkin_ws/src/detection_transforms/setup.py

from setuptools import find_packages, setup

setup(
    name='detection_transforms',
    version='0.1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=['numpy'],
)