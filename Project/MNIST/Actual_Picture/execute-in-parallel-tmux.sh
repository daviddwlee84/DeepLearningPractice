#!/bin/bash

tmux new-session -d 'python3 mnist_backward.py'
tmux split-window -h 'sleep 5s && python3 mnist_test.py'
tmux new-window 'mutt'
tmux -2 attach-session -d 
