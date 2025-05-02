from setuptools import setup, find_packages

setup(
    name="model_service",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "joblib~=1.4.2",
        "requests~=2.32.3",
        "Flask~=2.2.3"
    ],
    include_package_data=True,
    python_requires=">=3.8",
)