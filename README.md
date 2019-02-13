# Pyhawkes

it is a fork from linderman pyhawkes for further analysis on network

# Linux dependencies
 
## Arch
 
```bash
pacman -S gconf gsl npm 
```
## Centos 7x

```bash 
yum install libXScrnSaver gsl gsl-devel
```

### Nodejs-8x instalation
 
```bash  
curl --silent --location https://rpm.nodesource.com/setup_8.x | sudo bash -
yum install -y nodejs
```

### NodeJs dependencies 

```bash
npm install --unsafe-perm -g phantomjs-prebuilt
```

## Instaling orca without an x11 server running 

* download latest orca.App image from https://github.com/plotly/orca/releases
* create a bash script:

 ```bash
#!/bin/bash
xvfb-run -a /root/Downloads/orca-1.2.1-x86_64.AppImage "$@"
```
## Run Cisco Any connect and prevent to killing ssh session

```bash
route add -host 'client public ip' gw 'SERVER DEFUALT GATEWAY'
```

* cp it in /bin

# Compile setup.py file

Add 'build_ext --inplace' argument to setup.py 

# Python packages for build 

numpy, Cython