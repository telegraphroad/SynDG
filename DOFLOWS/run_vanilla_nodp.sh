#!/bin/bash


for clip in 0.0
do
    for lr in 1e-8 5e-8 1e-7 5e-7 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1. 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 6.0 8.0 10.0 20.0 30.0 50.0 100.0
    do
        python vanilla_vae.py --clip $clip --lr $lr
    done
done
