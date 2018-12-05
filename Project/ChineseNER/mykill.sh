#!/bin/bash
mykill() {
    ps -ef | grep "$1" | grep -v grep | awk '{print $2}' | xargs kill
    echo "process "$1" killed"
}