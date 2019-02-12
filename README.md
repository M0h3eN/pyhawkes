# pyhawkes

it is a fork from linderman pyhawkes for further analysis on network

# Linux(Arch) dependencies 

* pacman -S gconf gsl npm 

# Linux(Centos) dependencies
 
* yum install libXScrnSaver gsl gsl-devel

## nodejs instalation
 
* curl --silent --location https://rpm.nodesource.com/setup_8.x | sudo bash -
* yum install -y nodejs

# NodeJs dependencies 

*  npm install --unsafe-perm -g phantomjs-prebuilt

# instaling orca in centos without an x11 server running 

* download latest orca.App image from https://github.com/plotly/orca/releases
* create a bash script:

 ```bash
#!/bin/bash
xvfb-run -a /root/Downloads/orca-1.2.1-x86_64.AppImage "$@"
```
* cp it in /bin

# Compile setup.py file

add 'build_ext --inplace' argument to setup.py 

# Python packages for build 

numpy, Cython