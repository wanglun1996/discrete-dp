#!/bin/bash

python3.7 -m venv venv

git submodule init
git submodule update

mv discrete-gaussian-differential-privacy venv/lib64/python3.7/site-packages/dis_gauss
