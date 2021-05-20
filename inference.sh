#!/bin/bash
cd DCNv2_latest && ./make.sh
cd ../src && sudo python inference.py --config $1
