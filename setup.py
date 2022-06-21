from setuptools import find_packages, setup
setup(
    name='ml-library',
    packages=find_packages(include=['ml-library']),
    version='0.1.0',
    description='A Machine Learning library for learning purposes.',
    author='Liliana B',
    license='MIT',
    install_requires=['numpy'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)