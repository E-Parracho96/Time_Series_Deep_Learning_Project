# The Setup script is the centre of all activity in building, distributing, and installing modules.
# Responsible of creating my ML application as a package

from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."
def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requirements
    '''

    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
name='Deep_Learning_Time_Series_Project',
version= '0.0.1',
author='Emanuel',
author_email='emanuel_parracho@hotmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)