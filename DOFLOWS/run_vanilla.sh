#!/bin/bash


for clip in 0.8 3.0 2.0 1.5 1.0 0.9 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.05 0.03 0.01
do
    for lr in 0.0000001 50.0 0.000001 40.0 0.00001 30.0 0.0001 25.0 0.001 20.0 0.01 15.0 0.1 10.0 5.0 1.0
    do
        python vanilla_vae.py --clip $clip --lr $lr
    done
done
