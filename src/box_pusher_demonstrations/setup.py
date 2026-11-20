from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'box_pusher_demonstrations'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='V2 Box Pusher Project',
    maintainer_email='maintainer@example.com',
    description='Safe keyboard teleoperation and timestamped solo demonstration recorder for V2 box-pushing.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'keyboard_teleop = box_pusher_demonstrations.keyboard_teleop_node:main',
            'demonstration_recorder = box_pusher_demonstrations.demonstration_recorder_node:main',
            'dataset_inspect = box_pusher_demonstrations.dataset_inspect:main',
        ],
    },
)
