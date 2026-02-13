from setuptools import setup, find_packages
from pathlib import Path
direct = Path(__file__).parent
long_description = (direct / "README.md").read_text()

setup(
    name='ImgAlign',
    version='4.3',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    package_data={
    'ImgAlign.raft': ['raft-things.pth','LICENSE.txt'],
    },
    entry_points={
        'console_scripts': [
            'ImgAlign = ImgAlign.ImgAlign:__main__',
        ],
    },
    install_requires=[
        'mpl_interactions',
        'matplotlib',
        'numpy',
        'opencv-contrib-python',
        'scipy',
        'cupy',
        'scikit-learn',
        'python_color_transfer',
        'torch',
    ],
)
