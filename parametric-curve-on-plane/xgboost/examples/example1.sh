#!/bin/sh
EXM=example1
rm -f models/${EXM}.jl
rm -f predictions/${EXM}_pred.csv

#Archimedean spiral
FXT="0.1 * t * np.cos(t)"
FYT="0.1 * t * np.sin(t)"

TB=0.0
TE=20.0

python ../../common/pmc2t_gen.py \
  --dsout datasets/${EXM}_train.csv \
  --funcxt "$FXT" --funcyt "$FYT" \
  --tbegin $TB --tend $TE --tstep 0.01

python ../../../xgboost/fit_func_mimo.py \
  --trainds datasets/${EXM}_train.csv \
  --outputdim 2 \
  --modelout models/${EXM}.jl

python ../../common/pmc2t_gen.py --dsout datasets/${EXM}_test.csv --funcxt "$FXT" --funcyt "$FYT" --tbegin $TB --tend $TE --tstep 0.0475

python ../../../svr/predict_func_mimo.py \
 --model models/${EXM}.jl \
 --ds datasets/${EXM}_test.csv \
 --outputdim 2 \
 --measures mean_absolute_error mean_squared_error \
 --predictionout predictions/${EXM}_pred.csv

python ../../common/pmc2t_scatter.py --ds datasets/${EXM}_test.csv --prediction predictions/${EXM}_pred.csv
#python ../../common/pmc2t_scatter.py --ds datasets/${EXM}_test.csv --prediction predictions/${EXM}_pred.csv --savefig media/${EXM}.png
