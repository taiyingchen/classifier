from setuptools import setup, find_packages


with open('README.md') as file:
    readme = file.read()

with open('LICENSE') as file:
    license = file.read()

setup(
    name='classifier',
    version='1.0.0',
    description='Python implementation of Decision Tree and Naive Bayes classifiers',
    long_description=readme,
    author='Taiying Chen',
    author_email='taiying.tychen@gmail.com',
    url='https://github.com/taiyingchen/classifier',
    license='MIT',
    packages=find_packages(exclude=('tests', 'docs')),
    python_requires='>=3.6.0'
)
