#build a spark cluster
import json

def build_spark_cmd():
    JSONDC=json.JSONDecoder()
    p=JSONDC.decode(open('../spark.json','rU').read())
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
    print cmd
    cmd=p['ec2_dir']+'/'+cmd
    return cmd



