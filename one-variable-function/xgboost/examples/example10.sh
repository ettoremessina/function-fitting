#!/bin/sh
EXM=example10
rm -rf dumps/${EXM}
rm -rf logs/${EXM}
rm -rf snaps/${EXM}
rm -f predictions/${EXM}_pred.csv

FX="x * np.sin(1 / x)"
XB=-0.4
XE=0.4

python ../../common/fx_gen.py --dsout datasets/${EXM}_train.csv --funcx "$FX" --xbegin $XB --xend $XE --xstep 0.00031
python ../../common/fx_gen.py --dsout datasets/${EXM}_val.csv  --funcx "$FX" --xbegin $XB --xend $XE --xstep 0.00073

python ../../../xgboost/fit_func_miso.py \
 --trainds datasets/${EXM}_train.csv \
 --modelout models/${EXM}.jl \
 --valds datasets/${EXM}_val.csv

python ../../common/fx_gen.py --dsout datasets/${EXM}_test.csv  --funcx "$FX" --xbegin $XB --xend $XE --xstep 0.00073

python ../../../xgboost/predict_func_miso.py \
 --model models/${EXM}.jl \
 --ds datasets/${EXM}_test.csv \
 --predictionout predictions/${EXM}_pred.csv

python ../../common/fx_scatter.py \
  --ds datasets/${EXM}_test.csv \
  --prediction predictions/${EXM}_pred.csv \
  --title "XGBoost (all properties defaulted)" \
  --xlabel "x" \
  --ylabel "y=x sin(1/x)"

#python ../common/fx_scatter.py --ds datasets/${EXM}_test.csv --prediction predictions/${EXM}_pred.csv --savefig media/${EXM}.png

#python ../fx_video.py --modelsnap snaps/${EXM} --ds datasets/${EXM}_test.csv --savevideo media/${EXM}_test.gif
