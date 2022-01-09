#!/bin/bash
for x in sigmoid tanh sin cos relu leakyrelu elu selu celu silu mish gelu
do
  python main.py --name ${x} --activation ${x} --manual_seed 42 --output_path results/$x --max_steps 4001 --batch_size 1000 --num_layers 3 --num_hiddens 8 --eval_interval 200 --optimizer Adam --lr 0.01 --loss_fn MSE
  convert -delay 20 -loop 0 "results/${x}/*.png" "results/${x}_animation.gif"
done
