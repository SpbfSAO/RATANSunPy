from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ratansunpy",
    version="0.2.2",
    author=[
        "Igor Lysov <iilysov.sci@gmail.com>", 
        "Irina Knyazeva <iknyazeva@gmail.com>", 
        "Evgenii Kurochkin <k-u-r-o-k@yandex.ru>", 
        "Andrey Shendrik <ashend90@gmail.com>",
    ],
    description="RATAN-600 radioastronomical solar data and methods Python library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "astropy" >= "6.1.3",
        "numpy" >= "1.26.0",
        "pandas" >= "2.2.2",
        "python-dateutil" >= "2.8.2",
        "requests" >= "2.32.3",
        "scipy" == "1.13.1",
        "PyWavelets" == "1.7.0",
        "et-xmlfile" >= "1.1.0",
        "openpyxl" >= "3.1.5" 
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics"
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)