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
    version="0.5.3",

    long_description=long_description,
    long_description_content_type="text/markdown",

    packages=setuptools.find_packages(
        exclude=[
            "test",
            "examples"
        ]
    ),
    install_requires=[
        "lazy>=1.4",
        "networkx>=2.5.1",
        "numpy>=1.16.0",
        "pandas>=1.3.0",
        "plotly>=4.14.3",
        "scipy>=1.6.1",
        "tqdm>=4.47.0"
    ],
    package_data={
        "evobench.discrete.isg": ["data/*.txt"],
        "evobench.routing": ["data/*"],
        "evobench.continuous.cec2013lsgo": ["data/*.txt", "data/*.csv"]
    },
    include_package_data=True,
    tests_require=[
        "flake8>=4.0.1",
        "pytest>=5.4.3",
        "pytest-cov>=2.10.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
)
