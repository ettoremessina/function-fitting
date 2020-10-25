#!/bin/sh
EXM=example2
rm -f models/${EXM}.jl
rm -f predictions/${EXM}_pred.csv

#Twisted cubic
FXT="t"
FYT="t ** 2"
FZT="t ** 3"
NS="0.1 * np.random.normal(0.0, 1.0, size=sz)"

TB=0.0
TE=2.0

python ../../common/pmc3t_gen.py \
  --dsout datasets/${EXM}_train.csv \
  --funcxt "$FXT" --funcyt "$FYT" --funczt "$FZT" \
  --tbegin $TB --tend $TE --tstep 0.001 \
  --xnoise "$NS" --ynoise "$NS" --znoise "$NS"

python ../../../xgboost/fit_func_mimo.py \
  --trainds datasets/${EXM}_train.csv \
  --outputdim 3 \
  --modelout models/${EXM}.jl

python ../../common/pmc3t_gen.py \
  --dsout datasets/${EXM}_test.csv \
  --funcxt "$FXT" --funcyt "$FYT" --funczt "$FZT" \
  --tbegin $TB --tend $TE --tstep 0.00475

python ../../../xgboost/predict_func_mimo.py \
 --model models/${EXM}.jl \
 --ds datasets/${EXM}_test.csv \
 --outputdim 3 \
 --measures mean_absolute_error mean_squared_error \
 --predictionout predictions/${EXM}_pred.csv

python ../../common/pmc3t_scatter.py --ds datasets/${EXM}_test.csv --prediction predictions/${EXM}_pred.csv
#python ../../common/pmc3t_scatter.py --ds datasets/${EXM}_test.csv --prediction predictions/${EXM}_pred.csv --savefig media/${EXM}.png
