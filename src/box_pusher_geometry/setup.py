from setuptools import setup
from glob import glob
import os

package_name = 'box_pusher_geometry'

setup(
    name=package_name,
    version='0.2.6',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='V2 Box Pusher Project',
    maintainer_email='maintainer@example.com',
    description='RGB-D geometry estimation for tracked V2 orange boxes.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'orange_box_geometry = box_pusher_geometry.orange_box_geometry_node:main',
        ],
    },
)
