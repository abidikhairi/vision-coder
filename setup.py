from setuptools import setup, find_packages

setup(
    name='vision_coder',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # List your dependencies here
    ],
)
