from setuptools import setup
from glob import glob
import os

package_name = 'box_pusher_tracking'

setup(
    name=package_name,
    version='0.2.5',
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
    description='Multi-frame orange-box tracking for the V2 stereo robot.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'orange_box_tracker = box_pusher_tracking.orange_box_tracker_node:main',
        ],
    },
)
