#!/bin/bash

KENLM_PATH=$HOME/kenlm/build/bin
WDIR=$DP_DATA/ptb-sd-rnng

$KENLM_PATH/lmplz -o 5 < $WDIR/train.txt > $WDIR/train.arpa
$KENLM_PATH/build_binary $WDIR/train.arpa $WDIR/train.binary
$KENLM_PATH/query -v summary $WDIR/train.binary < $WDIR/dev.txt # -n
$KENLM_PATH/query -v summary $WDIR/train.binary < $WDIR/test.txt # -n

