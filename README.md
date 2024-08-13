This a readme file and process to go to install and activate the model

# Install MMAction2
1. prepare the environment

In the  ubuntu terminal you should use these to check the nvcc version and GCC version
```ubuntu terminal
# Check nvcc version`
!nvcc -V

# Check GCC version
!gcc --version
```

Output should look like this
please make sure that the cuda and ubuntu is compatibale with the mmaction

it should be showing something like this
```ubuntu terminal
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Feb__7_19:32:13_PST_2023
Cuda compilation tools, release 12.1, V12.1.66
Build cuda_12.1.r12.1/compiler.32415258_0

gcc (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0
Copyright (C) 2019 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is no
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```
2. follow the instruction on https://mmaction2.readthedocs.io/en/latest/get_started/installation.html to install everything
then use the command below
```ubuntu terminal
conda env list
conda activate open-mmlab
``` 

# Where to find the data
