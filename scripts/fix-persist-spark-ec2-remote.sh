#!/bin/sh
#launched from the master
SLAVES=`cat /root/spark-ec2/slaves`
FIX_SCRIPT=fix-persist-spark-ec2.sh
#fix yourself before helping others
/root/$FIX_SCRIPT

for slave in $SLAVES; do
    echo "fixing slave $slave"
    chmod u+x $FIX_SCRIPT
    scp /root/$FIX_SCRIPT $slave:
    ssh $slave /root/$FIX_SCRIPT
    ssh $slave rm /root/$FIX_SCRIPT
done
