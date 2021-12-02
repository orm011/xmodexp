from setuptools import find_packages
from distutils.core import setup
setup(
    name = 'xmodexp',
    version = '1.0.0',
    url = 'git@github.com:orm011/xmodexp.git',
    author = 'Oscar Moll',
    author_email = 'orm@csail.mit.edu',
    description = 'some experiments with cross modal embeddings',
    packages = find_packages(),
    install_requires = [],
)
