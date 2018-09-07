"""
Copyright 2017 NREL

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='ATom',
    version='1.0',
    description='Acoustic tomography of the atmosphere',
    long_description=long_description,
    url='https://github.nrel.gov/nhamilto/ATom',
    author='Nicholas Hamilton, NREL National Wind Technology Center',
    author_email='nicholas.hamilton@nrel.gov',
    license='Apache-2.0',
    classifiers=[  # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='wind turbine energy wake analysis nrel nwtc',
    packages=find_packages(here),
    install_requires=[
        'numpy >= 1.12.1',
        'scipy >= 0.19.1',
        'matplotlib >= 2.1.0',
        'pytest >= 3.3.1',
        'pandas >= 0.20'
    ],
    python_requires='>=3.6',
) #yapf: disable