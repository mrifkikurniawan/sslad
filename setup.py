#!/usr/bin/env python

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    
setup(
    name='online_continual_learning',
    version='1.0',
    description='Continual Learning Strategy for SSLAD Competition',
    author='Muhammad Rifki Kurniawan',
    author_email='mrifkikurniawan17@gmail.com',
    url='',
    install_requires=requirements,
    packages=find_packages(),
)
