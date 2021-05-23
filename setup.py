"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')
setup(
    name='aesahaettr',  # Required
    version='0.0.1',  # Required
    description='Inter-baseline foreground filtering tools',  # Optional
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://github.com/aewallwi/aesahaettr',  # Optional
    author='A. Ewall-Wice',  # Optional
    author_email='aaronew@berkeley.edu',  # Optional
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='21cm, cosmology, foregrounds, radio astronomy, cosmic dawn',
    package_dir={'aesahaettr': 'aesahaettr'},
    packages=['aesahaettr'],
    python_requires='>=3.6, <4',
    install_requires=['pyuvdata',
                      'numpy',
                      'tensorflow',
                      'numba',
                      'numba-scipy',
                      'scipy',                      
                      'uvtools @ git+git://github.com/HERA-Team/uvtools',
                      'pyuvsim @ git+git://github.com/RadioAstronomySoftwareGroup/pyuvsim',
                      'hera_sim @ git+git://github.com/HERA-Team/hera_sim',
                      ],
    exclude = ['tests'],
    zip_safe = False,
    )
