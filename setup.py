from setuptools import setup, find_packages

setup(
    name='DeepLearningUtils',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'keras',
        'pytest',
        # Add other common dependencies here
    ],
    extras_require={
        'torch': ['torch'],
        'tensorflow': ['tensorflow'],
    },
    entry_points={
        'console_scripts': [
            # Add command line scripts here if any
        ],
    },
)