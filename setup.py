from setuptools import setup, find_packages

setup(
    name='keras-cloud-ml-engine-tutorial',
    version='0.1',
    packages=find_packages(),  # ['trainer'],
    url='',
    license='MIT',
    author='anderson',
    author_email='a141890@gmail.com',
    description='',
    install_requires=[
        'keras',
        'h5py',
        'six'
    ]
)
