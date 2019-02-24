# Pyhawkes

it is a fork from linderman pyhawkes for further analysis on network

# Linux dependencies
 
## Arch
 
```bash
pacman -S gconf gsl npm 
```
## Centos 7x

```bash 
yum install libXScrnSaver gsl gsl-devel xorg-x11-server-Xvfb
```

## Run Cisco Any connect and prevent to killing ssh session

```bash
route add -host 'client public ip' gw 'SERVER DEFUALT GATEWAY'
```

## Latest version of python

```bash
yum install https://centos7.iuscommunity.org/ius-release.rpm
yum install python36u python36u-pip  python36u-devel
```

# Compile setup.py file

Add 'build_ext --inplace' argument to setup.py 

# Python packages for build 

numpy, Cython