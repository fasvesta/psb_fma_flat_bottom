#!/bin/bash

# export EOS_MGM_URL=root://eosuser.cern.ch

source /usr/local/xsuite/miniforge3/bin/activate xsuite
pwd

cp -r /afs/cern.ch/work/f/fasvesta/benchmarking_psb_3Qy/* .

python examplePSB_pic.py
