#!/bin/sh
#launch this from the master spark ec2 instance

#a script that fixes the persistent_hdfs being mounted on the
#non-persitent drives
#fixes itself (the master) and then if has "slaves" (or anything)
#for a parameter, it will copy itself to th slaves and run itself
#without a parameter to fix each slave

SLAVES=`cat /root/spark-ec2/slaves`
THIS_SCRIPT=$0
#fix yourself before helping others

mkdir /vol_backup
cp -r /vol/* /vol_backup/
rm -rf /vol
#does a force reformat of this, may want to make it less aggressive                                                                         
#unmount if it is mounted?
#umount /dev/xvdv
#format it
mkfs -t ext4 /dev/xvdv
 
XFS_MOUNT_OPTS=" defaults "
mkdir /vol
mount -o $XFS_MOUNT_OPTS /dev/xvdv /vol
cp -r /vol_backup/* /vol
rm -rf /vol_backup
chmod -R a+w /vol
echo "/dev/xvdv /vol    auto    $XFS_MOUNT_OPTS   0     0" >> /etc/fstab
#not sure if this is idempotent so don't do it
#mount -a                           
                                                                                                       
echo "Fixed persistent HDFS issue" > PER_MESSAGE

if [ x$1 = x"" ] ; then
    echo "no parameter, not recursivly fixing slaves, all done"
else
    echo "fixing slaves"

    for slave in $SLAVES; do
	echo "fixing slave $slave"
	chmod u+x $THIS_SCRIPT
	#copt it
	scp /root/$THIS_SCRIPT $slave:
	ssh $slave /root/$THIS_SCRIPT
	#run it
	ssh $slave rm /root/$THIS_SCRIPT
	#delete it from there
    done
fi

echo "done fixing persistent HDFS"
