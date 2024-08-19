#!/bin/bash
scrdir=`dirname $0`
cd "$scrdir"
python mqtt_micro_vadasr.py de_config.yml
