from setuptools import setup
from os import path

DIR = path.dirname(path.abspath(__file__))
INSTALL_PACKAGES = open(path.join(DIR, 'requirements.txt')).read().splitlines()

with open(path.join(DIR, 'README.md')) as f:
    README = f.read()

setup(
    name='back',
    packages=['back'],
    description='Backbone for PyTorch training loop',
    long_description=README,
    long_description_content_type='text/markdown',
    install_requires=INSTALL_PACKAGES,
    version='0.0.3',
    url='https://github.com/EvgenyKashin/backbone',
    author='Evgeny Kashin',
    author_email='kashinevge@gmail.com',
    keywords=['pytorch', 'deep-learning', 'training-loop', 'backbone'],
    license='MIT',
    python_requires='>=3.6.0'
)