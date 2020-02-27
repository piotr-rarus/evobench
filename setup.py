import setuptools


with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name='evobench',
    description='Benchmarks for large scale, model-based optimization.',

    author='Piotr Rarus',
    author_email='piotr.rarus@gmail.com',

    url='https://github.com/piotr-rarus/evobench',
    license='MIT',
    version='0.0.6',

    long_description=long_description,
    long_description_content_type='text/markdown',

    packages=setuptools.find_packages(
        exclude=[
            'test',
        ]
    ),
    install_requires=[
        'numpy',
        'sympy',
        'lazy',
        'tqdm'
    ],
    package_data={
        'evobench.discrete.isg': ['data/*.txt']
    },
    include_package_data=True,
    tests_require=[
        'pytest',
        'pytest-cov',
        'flake8',
        'pylint'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
)
