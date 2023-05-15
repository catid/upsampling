#!/bin/bash

mpirun --hostfile lightning_nodes.txt --display-map --map-by node -np 2 python lightning_train.py
