import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="cellcutter",
    version="0.1.1",
    description="Unsupervised deep learning for cell segmentation",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/jiyuuchc/cellcutter",
    author="Ji Yu",
    author_email="jyu@uchc.edu",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["cellcutter"],
    include_package_data=True,
    install_requires=[
      'numpy', 'scipy', 'scikit_learn', 'scikit_image', 'tensorflow'
    ]
)
