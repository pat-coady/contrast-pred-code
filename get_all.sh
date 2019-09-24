#!/bin/bash
DIR_NAME="replearn"
ssh $1 "tar zcvf payload.tar.gz -C $DIR_NAME/$DIR_NAME outputs"
scp $1:~/payload.tar.gz payload.tar.gz
tar --keep-newer-files -xvf payload.tar.gz -C $DIR_NAME/
rm payload.tar.gz
ssh $1 "tar zcvf payload.tar.gz -C $DIR_NAME/$DIR_NAME logs"
scp $1:~/payload.tar.gz payload.tar.gz
tar --keep-newer-files -xvf payload.tar.gz -C $DIR_NAME/
rm payload.tar.gz
