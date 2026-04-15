from setuptools import setup

package_name = 'race_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='V1NC3NT-CC',
    maintainer_email='V1NC3NT-CC@users.noreply.github.com',
    description='LiDAR local trajectory planner for F1TENTH racing simulation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'race_planner_node = race_planner.race_planner_node:main',
        ],
    },
)
