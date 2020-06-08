#!/bin/sh






gnome-terminal  --  bash  -c  "python3 ./auto_encoder.py torch traversal"

while [ 1 ]
do
    sleep 0.5
    wmctrl -c "Figure 1"
done