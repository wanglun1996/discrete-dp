#!/bin/bash

python3.7 -m venv venv

git submodule init
git submodule update

mv discrete-gaussian-differential-privacy venv/lib64/python3.7/site-packages/dis_gauss

git submodule init
git submodule update

wget -c https://leon.bottou.org/_media/projects/infimnist.tar.gz
tar -xzvf infimnist.tar.gz
mv ./infimnist/data ./infimnist_py
rm -rf infimnist
rm infimnist.tar.gz

cd infimnist_py
python setup.py build_ext -if
cd ..
