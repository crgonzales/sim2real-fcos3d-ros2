from setuptools import setup

package_name = 'fcos3d_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/detector.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Carlos Gonzales',
    maintainer_email='carlos.roberto.gonzales@gmail.com',
    description='FCOS3D monocular 3D detection as a ROS 2 node',
    license='MIT',
    entry_points={
        'console_scripts': [
            'detector_node = fcos3d_ros2.detector_node:main',
            'nuscenes_publisher = fcos3d_ros2.nuscenes_publisher:main',
        ],
    },
)
