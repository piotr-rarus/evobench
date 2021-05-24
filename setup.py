import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="evobench",
    description="Optimization benchmarks, both synthetic and practical.",

    author="Piotr Rarus",
    author_email="piotr.rarus@gmail.com",

    url="https://github.com/piotr-rarus/evobench",
    license="MIT",
    version="0.5.0",

    long_description=long_description,
    long_description_content_type="text/markdown",

    packages=setuptools.find_packages(
        exclude=[
            "test",
        ]
    ),
    install_requires=[
        "lazy>=1.4",
        "networkx>=2.5.1",
        "numpy>=1.16.0",
        "plotly>=4.14.3",
        "scikit-learn>=0.24.0",
        "scipy>=1.6.1",
        "tqdm>=4.47.0"
    ],
    package_data={
        "evobench.discrete.isg": ["data/*.txt"],
        "evobench.routing": ["data/*"]
    },
    include_package_data=True,
    tests_require=[
        "flake8>=3.8.3",
        "pytest>=5.4.3",
        "pytest-cov>=2.10.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
)
