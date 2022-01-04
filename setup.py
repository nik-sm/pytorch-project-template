from setuptools import setup, find_packages


with open("requirements.txt", "r") as f:
    required = f.readlines()

setup(
    name="pytorch_project_template",
    version="0.1",
    install_requires=required,
    packages=find_packages(),
)
