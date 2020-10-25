#!/bin/sh
EXM=example1_nusvr
rm -f models/${EXM}.jl
rm -f predictions/${EXM}_pred.csv

#Archimedean spiral on the space
FXT="0.1 * t * np.cos(t)"
FYT="0.1 * t * np.sin(t)"
FZT="t"

TB=0.0
TE=20.0

python ../../common/pmc3t_gen.py \
  --dsout datasets/${EXM}_train.csv \
  --funcxt "$FXT" --funcyt "$FYT" --funczt "$FZT" \
  --tbegin $TB --tend $TE --tstep 0.01

python ../../../svr/fit_func_nusvr.py \
  --trainds datasets/${EXM}_train.csv \
  --outputdim 3 \
  --modelout models/${EXM}.jl \
  --svrparams "'C': 20"

python ../../common/pmc3t_gen.py \
  --dsout datasets/${EXM}_test.csv \
  --funcxt "$FXT" --funcyt "$FYT" --funczt "$FZT" \
  --tbegin $TB --tend $TE --tstep 0.0475

python ../../../svr/predict_func.py \
 --model models/${EXM}.jl \
 --ds datasets/${EXM}_test.csv \
 --outputdim 3 \
 --measures mean_absolute_error mean_squared_error \
 --predictionout predictions/${EXM}_pred.csv

python ../../common/pmc3t_scatter.py --ds datasets/${EXM}_test.csv --prediction predictions/${EXM}_pred.csv
#python ../../common/pmc3t_scatter.py --ds datasets/${EXM}_test.csv --prediction predictions/${EXM}_pred.csv --savefig media/${EXM}.png
