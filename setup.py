import setuptools
from os import path

with open(path.join(path.abspath(path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(
    path.join(path.abspath(path.dirname(__file__)), 'requirements.txt'),
    encoding='utf-8',
) as f:
    requirements = f.read().splitlines()
    print(requirements)

setuptools.setup(  # type: ignore
    name='gca',
    version='0.1',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Marcel Robeer',
    packages=setuptools.find_packages(),  # type : ignore
    install_requires=requirements,
    python_requires='>=3.8',
)
