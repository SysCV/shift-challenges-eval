"""
# Q. How to install custom python pip packages?

# A. Uncomment the below code to install the custom python packages.

"""

import os
import subprocess
import sys
from pathlib import Path


def install(package):
    # Install a pip python package

    # Args:
    #     package ([str]): Package name with version

    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def install_local_package(folder_name):
    # Install a local python package

    # Args:
    #     folder_name ([str]): name of the folder placed in evaluation_script/

    subprocess.check_output(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            os.path.join(str(Path(__file__).parent.absolute()) + folder_name),
        ]
    )


# subprocess.check_call(["git", "--version"])
# subprocess.check_call(["git", "lfs", "--version"])
# subprocess.check_call(["git", "lfs", "pull"])

install("tqdm")
install("numpy==1.21")
# install("matplotlib==3.5.2")
install("nuscenes-devkit==1.1.10")
install("pyquaternion==0.9.9")
install("git+https://github.com/scalabel/scalabel.git@scalabel-evalAPI")
# install("Pillow==6.2.0")

print("============")
print("Install done")
print("============")

# show the version of the installed packages
subprocess.check_call(["pip", "freeze"])


from .main import evaluate
