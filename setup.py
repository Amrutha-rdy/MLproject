from setuptools import find_packages,setup
from typing import List

Hypen = '-e .'
def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path)  as file_object:
        requirements=file_object.readlines()

        requirements = [i.replace('\n','') for i in requirements]

        if Hypen in requirements:
            requirements.remove(Hypen)
    return requirements



setup(
name = 'mlproject',
version = '0.0.1',
author='amrutha',
author_email = 'amrdy9@gmail.com',
packages = find_packages(),
install_requires = get_requirements('requirements.txt')
)