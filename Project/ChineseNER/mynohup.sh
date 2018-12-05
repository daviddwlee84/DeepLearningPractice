#!/bin/bash
mynohup () {
    [[ "$1" = "" ]] && echo "usage: mynohup python_script" && return 0
    nohup python3 -u "$1" > log.txt 2>&1 < /dev/null &
}