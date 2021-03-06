# Install setuptools if not installed.
try:
    import setuptools
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import setup, find_packages


# read README as the long description
with open('README.md', 'r') as f:
    long_description = f.read()

with open('bayarea/requirements.txt') as f:
    requirements_lines = f.readlines()
install_requires = [r.strip() for r in requirements_lines]

setup(
    name='bayarea',
    version='0.1dev',
    description='Bay Area UrbanSim implementation',
    long_description=long_description,
    author='Urban Analytics Lab / UrbanSim Inc.',
    author_email='info@urbansim.com',
    license='BSD',
    url='https://github.com/ual/bayarea',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: BSD License'
    ],
    packages=find_packages(exclude=['*.tests']),
    install_require=install_requires
)
