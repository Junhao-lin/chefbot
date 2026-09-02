from setuptools import setup
import os
from glob import glob

package_name = 'rgbt_cooking_perception'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Shennongxi Cooking Robot Team',
    maintainer_email='user@example.com',
    description='RGBT Stream Processing and Realtime Wok Masking Package',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'stream_processor = rgbt_cooking_perception.stream_processor_node:main',
        ],
    },
)
