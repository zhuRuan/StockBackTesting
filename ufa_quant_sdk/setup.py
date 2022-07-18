from setuptools import setup, find_packages
PROJECT_NAME = 'ufa_quant_sdk'
VERSION = '0.0.1'

setup(name=PROJECT_NAME,
      version=VERSION,
      entry_points={
      },
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'simplejson==3.17.6',
          'requests==2.27.1',
      ],
      zip_safe=False
)
      
