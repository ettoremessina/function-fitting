#!/bin/sh
EXM=example1_lsvr
rm -f models/${EXM}.jl
rm -f predictions/${EXM}_pred.csv

FX="0.5*x**3 - 2*x**2 - 3*x - 1"
XB=-10.0
XE=10.0

python ../../common/fx_gen.py --dsout datasets/${EXM}_train.csv --funcx "$FX" --xbegin $XB --xend $XE --xstep 0.01

python ../../../svr/fit_func_nusvr.py \
  --trainds datasets/${EXM}_train.csv \
  --outputdim 1 \
  --modelout models/${EXM}.jl \
  --svrparams "'C': 100"

python ../../common/fx_gen.py --dsout datasets/${EXM}_test.csv  --funcx "$FX" --xbegin $XB --xend $XE --xstep 0.0475

python ../../../svr/predict_func.py \
 --model models/${EXM}.jl \
 --ds datasets/${EXM}_test.csv \
 --outputdim 1 \
 --measures mean_absolute_error mean_squared_error \
 --predictionout predictions/${EXM}_pred.csv

python ../../common/fx_scatter.py --ds datasets/${EXM}_test.csv --prediction predictions/${EXM}_pred.csv
#python ../../common/fx_scatter.py --ds datasets/${EXM}_test.csv --prediction predictions/${EXM}_pred.csv --savefig media/${EXM}.png
