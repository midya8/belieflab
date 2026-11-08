import argparse,json
from .core import solve,belief_update
from .demo import demo
from .open_loop import solve_open_loop,compare_feedback
from .feedback_demo import feedback_demo


def pairs(items):
    result={}
    for k,v in items:
        if k in result:raise ValueError('duplicate JSON key: '+k)
        result[k]=v
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description='Exact two-state finite-horizon POMDP policy envelopes');s=p.add_subparsers(dest='command',required=True);s.add_parser('demo');s.add_parser('feedback-demo')
    for name in ['solve','update','open-loop','compare']:
        a=s.add_parser(name);a.add_argument('model');a.add_argument('--belief',required=True)
        if name in ['solve','open-loop','compare']:
            a.add_argument('--horizon',type=int,default=2)
            if name!='open-loop':a.add_argument('--candidate-limit',type=int,default=4096)
        else:a.add_argument('--action',type=int,required=True);a.add_argument('--observation',type=int,required=True)
    args=p.parse_args(argv)
    try:
        if args.command=='demo':out=demo()
        elif args.command=='feedback-demo':out=feedback_demo()
        else:
            with open(args.model,'rb') as f:raw=f.read(262145)
            if len(raw)>262144:raise ValueError('model JSON exceeds 256 KiB')
            model=json.loads(raw,object_pairs_hook=pairs);belief=json.loads(args.belief)
            if args.command=='solve':out=solve(model,belief,horizon=args.horizon,candidate_limit=args.candidate_limit)
            elif args.command=='open-loop':out=solve_open_loop(model,belief,horizon=args.horizon)
            elif args.command=='compare':out=compare_feedback(model,belief,horizon=args.horizon,candidate_limit=args.candidate_limit)
            else:out=belief_update(model,belief,args.action,args.observation)
        print(json.dumps(out,indent=2,allow_nan=False))
    except (ValueError,TypeError,OSError,OverflowError) as e:p.exit(2,'error: '+str(e)+'\n')

if __name__=='__main__':main()
