from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='bayesian_deep_learning',
    version='0.0.2',
    author='Ravid Shwartz-Ziv',
    description='Bayesian deep learning ',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/ravidziv/self-supervised-learning',
    project_urls={
        "Bug Tracker": "https://github.com/ravidziv/self-supervised-learning/issues"
    },
    license='MIT',
    packages=find_packages(),
    install_requires=['tensorflow', 'tensorflow-addons'],
)
