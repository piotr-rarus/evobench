import setuptools


with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name='evobench',
    description='Benchmarks for model-based optimization.',

    author='Piotr Rarus',
    author_email='piotr.rarus@gmail.com',

    url='https://github.com/piotr-rarus/evobench',
    license='MIT',
    version='0.2.2',

    long_description=long_description,
    long_description_content_type='text/markdown',

    packages=setuptools.find_packages(
        exclude=[
            'test',
        ]
    ),
    install_requires=[
        'lazy==1.4',
        'numpy==1.18.4',
        'tqdm==4.46.0'
    ],
    package_data={
        'evobench.discrete.isg': ['data/*.txt']
    },
    include_package_data=True,
    tests_require=[
        'flake8==3.7.9',
        'pytest==5.3.5',
        'pytest-cov==2.8.1'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
)
