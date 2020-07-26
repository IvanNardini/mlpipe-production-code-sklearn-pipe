#!/bin/bash

cd ./Pipeline

for script in $*; do
    if [ $script == 'train' ]; then
        python3 train.py
    fi
done