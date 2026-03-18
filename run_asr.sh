#!/bin/bash
scrdir=`dirname $0`
cd "$scrdir"
#python mqtt_micro_vadasr.py de_config.yml
/home/rob/.local/bin/uv run python transcriptor.py -c config_de.yml
