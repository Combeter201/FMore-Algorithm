from distutils.core import setup

setup(name='fmore',
      version='0.7',
      description='Implementation of FMore algorithm',
      package_dir={'': 'src'},
      packages=['fmore','fmore.node','fmore.aggregator.strategy'],
      install_requires=['tensorflow', 
                        'flwr']
      )