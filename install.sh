#!/usr/bin/env sh
HOME=`pwd`

# Chamfer Distance
cd extensions/chamfer_dist
# python setup.py install --user
python setup.py build_ext --inplace



# EMD
cd $HOME/extensions/emd
# python setup.py install --user
python setup.py build_ext --inplace

