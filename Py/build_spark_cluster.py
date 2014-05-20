#build a spark cluster
import json

def build_spark_cmd(spark_file="../myspark.json"):
    JSONDC=json.JSONDecoder()
    p=JSONDC.decode(open(spark_file,'rU').read())
    cmd="spark-ec2 -k %s -i %s.pem -t %s"%(p['pem'],p['pem'],p['type'])
    cmd=cmd+" -u %s -s %s -r %s"%(p['user'],p['n_slaves'],p['region'])
    cmd=cmd+" --ebs-vol-size %s"%p['diskGB']
    print p['spot_price'].__class__
    if p['spot_price'].__class__ in [int,float]:
        print "Trying for spot price %s on %s in region %s"%(p['spot_price'],p['type'],p['region'])
        spot=float(p['spot_price'])
        if spot > 0 and spot < 1.0:
            cmd=cmd+" --spot-price %s"%spot
        else:
            raise(ValueError,"spot price must be between 0 and 1 or a string")
    else:
        print "not using spot price, not a number"
        pass
             
    cmd=cmd +" launch %s"%p['name']
    cmd=p['ec2_dir']+'/'+cmd
    print "command is:"
    print cmd
    return cmd

def run_in_shell(cmd):
    import os
    full_cmd="source ~/creds.sh ; %s"%cmd
    os.system(full_cmd)

if __name__ == "__main__":
    import sys
    args=sys.argv
    if len(args) <= 1:
        cmd=build_spark_cmd()
    else :
        spark_file=sys.argv[1] 
        cmd=build_spark_cmd(spark_file)
    run_in_shell(cmd)
