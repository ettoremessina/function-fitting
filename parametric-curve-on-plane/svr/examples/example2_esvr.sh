#!/bin/sh
EXM=example2_esvr
rm -f models/${EXM}.jl
rm -f predictions/${EXM}_pred.csv

#Bernoulli's spiral
FXT="np.exp(0.1 * t) * np.cos(t)"
FYT="np.exp(0.1 * t) * np.sin(t)"
NS="2.5 * np.random.normal(0.0, 1.0, size=sz)"

TB=0.0
TE=20.0

python ../../common/pmc2t_gen.py \
  --dsout datasets/${EXM}_train.csv \
  --funcxt "$FXT" --funcyt "$FYT" \
  --tbegin $TB --tend $TE --tstep 0.01 \
  --noisex "$NS" --noisey "$NS"

python ../../../svr/fit_func_esvr.py \
  --trainds datasets/${EXM}_train.csv \
  --outputdim 2 \
  --modelout models/${EXM}.jl \
  --svrparams "'kernel': 'rbf', 'C': 100, 'gamma': 0.1, 'epsilon': 0.1"

python ../../common/pmc2t_gen.py --dsout datasets/${EXM}_test.csv --funcxt "$FXT" --funcyt "$FYT" --tbegin $TB --tend $TE --tstep 0.0475

python ../../../svr/predict_func.py \
 --model models/${EXM}.jl \
 --ds datasets/${EXM}_test.csv \
 --outputdim 2 \
 --measures mean_absolute_error mean_squared_error \
 --predictionout predictions/${EXM}_pred.csv

python ../../common/pmc2t_scatter.py --ds datasets/${EXM}_test.csv --prediction predictions/${EXM}_pred.csv
#python ../../common/pmc2t_scatter.py --ds datasets/${EXM}_test.csv --prediction predictions/${EXM}_pred.csv --savefig media/${EXM}.png
