#!/bin/bash -l

for seed in 0 #1 2 3
do
    for data_subset in 1000 10000 100000 200000
    do
        for name in fcn_pickplace fcn_shift cfm autoencoder visual_forward
        do
            python3 "./run_methods/run_train_$name.py" --seed "$seed" --data_subset "$data_subset"
        done
    done
done


for dir in ./out/*/
do
  dir=${dir%*/}
  python3 run_evaluation.py "./out/${dir##*/}"
done

