#!/bin/bash
for nl in 18 22 25 28; do
    for w in  4096 2048 1024 768 512 378 256 192 128; do
        for ml in 6 5 4 3 2 1; do
            for lr in 0.000001 0.001; do
                python normalizing-flows/examples/tabular_test_hyp_onlynum.py $nl $w $ml $lr
            done
        done
    done
done