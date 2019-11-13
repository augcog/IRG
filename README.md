# IRG

IRG is an open-source python library for supporting an autonomous driving competition organized by the University of California, Berkeley, called ROAR. The library is currently developed and managed by a team of researchers at UCB.

Please review the license agreement and the third-party licenses before the installation of the software.

## Installation Guide on PC

1. Windows

If your PC includes an NVidia GPU that supports the latest CUDA SDK (e.g., 10.1), you should first install (1) the latest NVidia GPU driver; (2) CUDA SDK; and (3) cuDNN library.

- First, make sure miniconda Python 3.7 is installed on the system. Verify that conda has the latest updates
```bash
conda update -n base -c defaults conda
```

- In the base IRG project folder, create the IRG anaconda environment
```bash
conda env create -f install\envs\windows.yml
conda activate irg
pip install -e .[pc]
```

- Next, install tensorflow with GPU support
```bash
conda install tensorflow-gpu
```

- Finally, create a local executable directory
```bash
irg createcar --path ~\ROAR
```

2. Linux

If your PC includes an NVidia GPU that supports the latest CUDA SDK (e.g., 10.1), you should first install the latest NVidia GPU driver from the NVidia website.

- First, make sure miniconda Python 3.7 is installed on the system. Verify that conda has the latest updates
```bash
conda update -n base -c defaults conda
```

- In the base IRG project folder, create the IRG anaconda environment
```bash
conda env create -f install/envs/ubuntu.yml
conda activate irg
pip install -e .[pc]
```

- If the Linux machine supports NVidia CUDA, anaconda will  install tensorflow with GPU support
```bash
conda install tensorflow-gpu
```

- Finally, create a local executable directory
```bash
irg createcar --path ~/ROAR
```

3. Mac OS X

- First, make sure miniconda Python 3.7 is installed on the system. Verify that conda has the latest updates
```bash
conda update -n base -c defaults conda
```

- In the base IRG project folder, create the IRG anaconda environment
```bash
conda env create -f install/envs/mac.yml
conda activate irg
pip install -e .[pc]
```

- Finally, create a local executable directory
```bash
irg createcar --path ~/ROAR
```

## How to re-install the package

When you previously installed a version of the IRG package, after pulling the latest version, a re-install is needed to update the changes in local binaries. 

- Under the root IRG directory, re-install the package on Jetson Nano
```bash
pip install -e .[nano]
```
On PC, run
```bash
pip install -e .[pc]
```

# How to calibrate the vehicle

Every vehicle before being modified with AI components must be calibrated with respect to its chassis

- By default, the vehicle should choose a LiPo battery. One should verify that the vehicle ESC has the LiPo low-voltage protection function activated if this function is available. Low-voltage LiPo batteries can be permanently damaged or even explode when used improperly.

- Some RC cars have different control modes in their ESC, such as training versus racing mode. One should verify the ESC is in the correct mode such that later PWM commands can be fully executed.

- The vehicle needs to be driven first using the factory radio controller. If the vehicle with neutral steering biases towards to the left or to the right, then such bias must be corrected by adjusting the steering counter-bias on the radio controller. 

Finally, after the AI components are installed, one needs to calibrate the range of the PWM signals separately for steering and thruttle. Assuming the PCA servo board is connected to the Jetson Nano GPIO on bus 1 I2C (alternative is bus 0), then the calibration is via the command for calibrating thruttle
```bash
irg calibrate --channel 0 --bus=1
```
and for calibrating steering
```bash
irg calibrate --channel 1 --bus=1
```