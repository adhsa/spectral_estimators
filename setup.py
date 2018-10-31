from setuptools import setup


def requirements():
    with open('requirements.txt', 'r') as f:
        modules = [module.strip() for module in f if module.strip()]
    return modules


def readme():
    with open('README.md', 'r') as f:
        text = f.read()
    return text


setup(
    name='spectral_estimators',
    author='Adham Sakhnini',
    author_email='asatna13@gmail.com',
    description='A library of high resolution spectral estimators and utilities.',
    long_description=readme(),
    install_requires=requirements(),
    version='0.0.0',
    url='https://github.com/adhsa/spectral_estimators',
    license='MIT',
    packages=['spectral_estimators'],
)
