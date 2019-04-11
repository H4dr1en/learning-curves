import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="learning-curves",
    version="0.2.2",
    author="H4dr1en",
    author_email="h4dr1en@pm.me",
    description="Python module allowing to easily calculate and plot the learning curve of a machine learning model and find the maximum expected accuracy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/H4dr1en/learning-curves",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research'
    ],
    license='MIT',
    download_url = 'https://github.com/H4dr1en/learning-curves/archive/0.2.2.tar.gz',
    keywords = ['Learning', 'curve', 'machine', 'learning', 'saturation', 'accuracy'],
    install_requires=[
        'dill',
        'numpy',
        'sklearn',
        'scipy',
        'matplotlib'
    ],
)