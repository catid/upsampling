#!/bin/bash

deepspeed -H hostfile train.py --deepspeed $@
