#!/bin/bash

SRILM_PATH=$HOME/srilm/lm/bin/i686-m64
WDIR=$DP_DATA/ptb-sd-working #-rnng

$SRILM_PATH/ngram-count -text $WDIR/train.txt -order 5 -kndiscount -interpolate -gt3min 0 -gt4min 0 -gt5min 0 -unk -lm $WDIR/train.srilm.arpa
$SRILM_PATH/ngram -ppl $WDIR/dev.txt -order 5 -unk -lm $WDIR/train.srilm.arpa
$SRILM_PATH/ngram -ppl $WDIR/test.txt -order 5 -unk -lm $WDIR/train.srilm.arpa

