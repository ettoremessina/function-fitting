#!/bin/sh
EXM=example10
rm -rf dumps/${EXM}
rm -rf logs/${EXM}
rm -rf snaps/${EXM}
rm -f predictions/${EXM}_pred.csv

FXY="2 * np.sin(x)**2 / np.sqrt(1 + y**2)"
XB=-3.0
XE=3.0
YB=-3.0
YE=3.0

python ../../common/fxy_gen.py --dsout datasets/${EXM}_train.csv --funcxy "$FXY" --xbegin $XB --xend $XE --ybegin $YB --yend $YE --xstep 0.05 --ystep 0.1
python ../../../xgboost/fit_func_miso.py \
  --trainds datasets/${EXM}_train.csv \
  --modelout models/${EXM}.jl

python ../../common/fxy_gen.py --dsout datasets/${EXM}_test.csv  --funcxy "$FXY" --xbegin $XB --xend $XE --ybegin $YB --yend $YE  --xstep 0.0875 --ystep 0.5
python ../../../xgboost/predict_func_miso.py \
 --model models/${EXM}.jl \
 --ds datasets/${EXM}_test.csv \
 --predictionout predictions/${EXM}_pred.csv

python ../../common/fxy_scatter.py \
  --ds datasets/${EXM}_test.csv \
  --prediction predictions/${EXM}_pred.csv \
  --title "XGBoost (all properties defaulted)" \
  --xlabel "x" \
  --ylabel "y" \
  --zlabel "z=2 sin(x)^2 / sqrt(1+y^2)"
#python ../fxy_scatter.py --ds datasets/${EXM}_test.csv --prediction predictions/${EXM}_pred.csv --savefig media/${EXM}.png

#python ../fxy_video.py --modelsnap snaps/${EXM} --ds datasets/${EXM}_test.csv --savevideo media/${EXM}_test.gif
