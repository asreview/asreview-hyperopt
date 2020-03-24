# Always prefer setuptools over distutils
from setuptools import setup, find_namespace_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Extract version from cbsodata.py
for line in open(path.join("asreviewcontrib", "hyperopt", "__init__.py")):
    if line.startswith('__version__'):
        exec(line)
        break

DEPS = {
    'mpi': ['mpi4py'],
}

DEPS['all'] = DEPS['mpi']


setup(
    name='asreview-hyperopt',
    version=__version__,  # noqa
    description='Hyper parameter optimization extension for ASReview',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/asreview/asreview-hyperopt',
    author='Utrecht University',
    author_email='asreview@uu.nl',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Pick your license as you wish
        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='asreview plot hyperopt optimization',
    packages=find_namespace_packages(include=['asreviewcontrib.*']),
    namespace_package=["asreview"],
    install_requires=[
        "asreview>=0.7.0", "numpy", "tqdm", "hyperopt", "sklearn",
    ],

    extras_require=DEPS,

    entry_points={
        "asreview.entry_points": [
            "hyper-active = asreviewcontrib.hyperopt:HyperActiveEntryPoint",
            "hyper-passive = asreviewcontrib.hyperopt:HyperPassiveEntryPoint",  #noqa
            "hyper-cluster = asreviewcontrib.hyperopt:HyperClusterEntryPoint",  #noqa
            "show = asreviewcontrib.hyperopt:ShowTrialsEntryPoint",
            "create-config = asreviewcontrib.hyperopt:CreateConfigEntryPoint",  #noqa
        ]

    },

    project_urls={
        'Bug Reports':
            "https://github.com/asreview/asreview-hyperopt/issues",
        'Source':
            "https://github.com/asreview/asreview-hyperopt",
    },
)
