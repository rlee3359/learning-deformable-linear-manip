#!/bin/bash

for dir in ./out/*/
do 
  dir=${dir%*/}
  python3 run_evaluation.py "./out/${dir##*/}"
done

