#!/bin/bash

deepspeed train.py --deepspeed --deepspeed_config deepspeed_config.json $@
