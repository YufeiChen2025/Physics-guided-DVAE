"""Retain tuned adapted P3 DVAE, and use direct concatenation of observed new7.
Baseline weights and primary objective are reused; Stage2 schedule follows 2.0.
"""
from pathlib import Path
import shutil,math
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,sys,json,hashlib,time,traceback,statistics
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal,kl_divergence
AP=argparse.ArgumentParser();AP.add_argument('--version',required=True,choices=['1.1.0','1.1.1']);AP.add_argument('--seed',type=int,default=42);AP.add_argument('--role',default='stage1',choices=['prepare','preflight','stage1','continue','evidence','rule','chain','full','test']);ARGS=AP.parse_args()
HOME = Path(__file__).resolve().parent
BASE_CODE = HOME.parent / ('experiments_3w_v' + ARGS.version.replace('.', '_'))
sys.path.insert(0, str(BASE_CODE / 'src'))
from common import ROOT as BASE
if 'DVAE_V111_WORK' in os.environ:
    default_out = Path(os.environ['DVAE_V111_WORK']).resolve() / 'adapted'
    OUT = Path(os.environ.get('DVAE_V111_ADAPTED', default_out)).resolve()
else:
    OUT = HOME / ARGS.version
from models import Diagnoser,MaskedChain,masked_average
from train import seed_everything,atomic_torch_save,rng_state,restore_rng,classification_metrics
from torch_dvae.DVAE import DVAE
from torch_dvae.encoder_models import RNNEncoder
from torch_dvae.inference_models import RNNInference
from torch_dvae.initializers import ControlInitializer
from torch_dvae.measurement_models import MLPMeasure
from torch_dvae.transition_models import MLPTransition
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu');SENSOR=[1,3,2,4]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def raw_features(part):
 assert part in ['train','val','test'];x=np.load(BASE/'cache'/part/'x.npy',mmap_mode='r');m=np.load(BASE/'cache'/part/'mask.npy',mmap_mode='r');a=np.zeros((len(x),7));v=np.zeros_like(a,dtype=bool)
 for i in range(0,len(x),512):
  xx=np.array(x[i:i+512][:,:,SENSOR],dtype=float);mm=np.array(m[i:i+512][:,:,SENSOR],dtype=bool);f=mm[:,:32];l=mm[:,-32:];nf=f.sum(1);nl=l.sum(1);ok=(nf>0)&(nl>0);a[i:i+len(xx),:4]=((xx[:,-32:]*l).sum(1)/np.maximum(nl,1)-(xx[:,:32]*f).sum(1)/np.maximum(nf,1))*ok;v[i:i+len(xx),:4]=ok
  pair=mm[:,1:]&mm[:,:-1];diff=np.diff(xx,axis=1);cnt=pair.sum(1)
  for j,c in enumerate([0,2]):a[i:i+len(xx),4+j]=np.sqrt((diff[:,:,c]**2*pair[:,:,c]).sum(1)/np.maximum(cnt[:,c],1));v[i:i+len(xx),4+j]=cnt[:,c]>0
  joint=mm[:,:,0]&mm[:,:,1];n=joint.sum(1);p=xx[:,:,0];q=xx[:,:,1];dp=(p-(p*joint).sum(1)[:,None]/np.maximum(n,1)[:,None])*joint;dq=(q-(q*joint).sum(1)[:,None]/np.maximum(n,1)[:,None])*joint;vp=(dp**2).sum(1);vq=(dq**2).sum(1);ok=(n>=2)&(vp>0)&(vq>0);a[i:i+len(xx),6]=np.where(ok,(dp*dq).sum(1)/np.sqrt(np.maximum(vp*vq,1e-300)),0).clip(-1,1);v[i:i+len(xx),6]=ok
 return a,v

def transform(a,v,tf):
 out=np.zeros_like(a,dtype=np.float32)
 for j,s in enumerate(tf['dims']):
  if len(s['values'])>1:out[:,j]=((2*np.interp(a[:,j],s['values'],s['probabilities'],left=0,right=1)-1)-s['center'])*v[:,j]
 return out

def prepare():
 p=OUT/'configs/transform.json'
 if not p.exists():
  a,v=raw_features('train');y=np.load(BASE/'cache/train/index.npy')[:,2];rng=np.random.default_rng(20261110);ids=[]
  for c in range(9):
   pool=np.flatnonzero(y==c);ids.extend(rng.choice(pool,min(len(pool),max(1,round(len(pool)*80000/len(y)))),False))
  ids=np.array(ids,dtype=int);dims=[];qs=np.linspace(0,1,257)
  for j in range(7):
   z=a[ids,j][v[ids,j]]
   if not len(z):dims.append({'values':[0.],'probabilities':[.5],'center':0.});continue
   knots=np.quantile(z,qs);values,inv=np.unique(knots,return_inverse=True);probs=np.bincount(inv,weights=qs)/np.bincount(inv);center=float((2*np.interp(z,values,probs,left=0,right=1)-1).mean()) if len(values)>1 else 0.;dims.append({'values':values.tolist(),'probabilities':probs.tolist(),'center':center})
  tf={'fit_part':'train only','dims':dims,'normalization_sha256':sha(BASE/'configs/normalization.json')};save(p,tf);np.save(OUT/'cache/train_e.npy',transform(a,v,tf));np.save(OUT/'cache/train_valid.npy',v)
 tf=read(p)
 for part in ['train','val']:
  dest=OUT/'cache'/f'{part}_e.npy'
  if not dest.exists():a,v=raw_features(part);np.save(dest,transform(a,v,tf));np.save(OUT/'cache'/f'{part}_valid.npy',v)
 # Diagonal Gaussian class prototype yields a train-derived linear energy.
 e=np.load(OUT/'cache/train_e.npy');v=np.load(OUT/'cache/train_valid.npy');y=np.load(BASE/'cache/train/index.npy')[:,2];x=np.tanh(e);mean=np.array([x[v[:,j],j].mean() if v[:,j].any() else 0 for j in range(7)]);var=np.array([max(float(x[v[:,j],j].var()),1e-4) if v[:,j].any() else 1. for j in range(7)]);centers=np.stack([np.array([x[(y==c)&v[:,j],j].mean() if ((y==c)&v[:,j]).any() else mean[j] for j in range(7)]) for c in range(9)]);centered=centers-mean;w=centered/var;b=-.5*(centered**2/var).sum(1)-w@mean;save(OUT/'configs/new7_empirical_prior.json',{'weight':w.tolist(),'bias':b.tolist(),'fit_on':'train only','meaning':'train-derived class response prior; NOT original valve/BSW physics prior','classes':9})
 save(OUT/'prepared.json',{'status':'ready','no_test_read':True,'version':ARGS.version})

class Model(nn.Module):
 def __init__(self,stage,role,seed):
  super().__init__();self.stage=stage;self.role=role;self.with_e=stage==2 and role!='continue';self.with_r=role in ['rule','full'];self.with_c=role in ['chain','full'];d=14
  self.dvae=DVAE(RNNInference(8,64,64,64),MLPTransition(8,64,0,64),MLPMeasure(7,8,64,64,64),RNNEncoder(d,64),RNNEncoder(d,64),ControlInitializer(64,8,0,d))
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(seed+211);self.classifier=nn.Sequential(nn.Linear(31 if self.with_e else 24,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,9))
  self.register_buffer('class_weights',torch.ones(9))
  self.gain_raw=nn.Parameter(torch.full((7,),float(np.log(np.expm1(1.)))))
  self.rule_head=nn.Linear(7,9);prior=read(OUT/'configs/new7_empirical_prior.json')
  with torch.no_grad():self.rule_head.weight.copy_(torch.tensor(prior['weight']));self.rule_head.bias.copy_(torch.tensor(prior['bias']))
  with torch.random.fork_rng(devices=[]):torch.manual_seed(seed+307);self.chain=MaskedChain(read(BASE/'configs/normalization.json')['features'],8,64)
 def evidence(self,e):return torch.tanh(F.softplus(self.gain_raw)*e)
 def logits(self,z,e):
  h=torch.cat([z.mean(1),z.std(1,unbiased=False),z[:,-1]-z[:,0]],-1)
  return self.classifier(torch.cat([h,self.evidence(e)],-1) if self.with_e else h)
 def encode(self,x,m):
  inp=torch.cat([x,m.to(x.dtype)],-1);return self.dvae.inference_func(inp,inp),inp
 def predict(self,x,m,e):return self.logits(self.encode(x,m)[0][0].loc,e)
 def loss(self,x,m,y,e):
  (post,yhist,_,_,xe),inp=self.encode(x,m);post=Normal(post.loc,post.scale.clamp(.02,5));z=post.rsample();ir=self.dvae.init_net(inp,inp);init=Normal(ir.loc,ir.scale.clamp(.02,5)).rsample();pr=self.dvae.get_priors(init,z,yhist,xe);prior=Normal(pr.loc,pr.scale.clamp(.02,5));obs=self.dvae.measure.get_dist(z,yhist,xe);nll=masked_average(.5*(obs.loc-x).square()+.5*math.log(2*math.pi),m);kl=kl_divergence(post,prior).mean();ce=F.cross_entropy(self.logits(post.loc,e),y,weight=self.class_weights);rule=F.cross_entropy(self.rule_head(self.evidence(e)),y) if self.with_r else ce.new_zeros(());chain=ce.new_zeros(())
  if self.with_c:prefix=self.encode(x[:,:64],m[:,:64])[0][0].loc;chain,_=self.chain(x,m,prefix)
  return .1*nll+.001*kl+2*ce+.0005*rule+.001*chain,{'ce':ce,'nll':nll,'kl':kl,'rule':rule,'chain':chain}

def load_data():
 d={}
 for part in ['train','val']:
  idx=np.load(BASE/'cache'/part/'index.npy');d[part]={'x':torch.tensor(np.load(BASE/'cache'/part/'x.npy'),device=DEVICE),'m':torch.tensor(np.load(BASE/'cache'/part/'mask.npy'),device=DEVICE,dtype=torch.bool),'y':torch.tensor(idx[:,2].astype(np.int64),device=DEVICE),'e':torch.tensor(np.load(OUT/'cache'/f'{part}_e.npy'),device=DEVICE)}
 return d
@torch.no_grad()
def evaluate(model,d):
 model.eval();out=[]
 for i in range(0,len(d['y']),256):out.append(model.predict(d['x'][i:i+256],d['m'][i:i+256],d['e'][i:i+256]).cpu().numpy())
 logits=np.concatenate(out);assert np.isfinite(logits).all();return classification_metrics(d['y'].cpu().numpy(),logits.argmax(1)),logits

def initialise_stage2(seed, role):
 seed_everything(seed)
 model = Model(2, role, seed).to(DEVICE)
 baseline = torch.load(OUT / 'runs' / f'seed{seed}_stage1' / 'best.pt',
                       map_location=DEVICE)['model']
 model.dvae.load_state_dict({name.removeprefix('dvae.'): value for name, value in baseline.items()
                             if name.startswith('dvae.')})
 with torch.no_grad():
  model.classifier[0].weight[:, :24].copy_(baseline['classifier.0.weight'])
  model.class_weights.copy_(baseline['class_weights'])
  model.classifier[0].bias.copy_(baseline['classifier.0.bias'])
  model.classifier[3].weight.copy_(baseline['classifier.3.weight'])
  model.classifier[3].bias.copy_(baseline['classifier.3.bias'])
 return model

def train(max_epochs=10):
 seed=ARGS.seed;role=ARGS.role;stage=2;folder=OUT/'runs'/f'seed{seed}_{role}';folder.mkdir(exist_ok=True)
 if (folder/'result.json').exists():return
 model = initialise_stage2(seed, role)
 lr=.001 if stage==1 else .0003;epochs=max_epochs;patience=6 if stage==1 else 4;opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-5);d=load_data();begin=0;best=-1.;stale=0;history=[]
 if (folder/'last.pt').exists():
  c=torch.load(folder/'last.pt',map_location=DEVICE);model.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);begin=c['next_epoch'];best=c['best'];stale=c['stale'];history=c['history'];restore_rng(c['rng'])
 else:atomic_torch_save(folder/'initial.pt',{'model':model.state_dict()})
 for epoch in range(begin,epochs):
  model.train();order=np.random.default_rng(seed*10000+epoch).permutation(len(d['train']['y']));tic=time.monotonic()
  for step,start in enumerate(range(0,len(order),32)):
   ix=order[start:start+32];torch.manual_seed(seed*1000000+epoch*10000+step);a=d['train'];loss,parts=model.loss(a['x'][ix],a['m'][ix],a['y'][ix],a['e'][ix]);assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);opt.step()
   if step%1000==0:save(folder/'progress.json',{'epoch':epoch+1,'step':step,'steps':(len(order)+31)//32,'best':best})
  metric,logits=evaluate(model,d['val']);history.append({'epoch':epoch+1,'validation_p':float(np.mean([v['precision'] for v in metric['per_class'].values()])),'validation_r':metric['balanced_accuracy'],'validation_f1':metric['macro_f1'],'seconds':time.monotonic()-tic})
  if metric['macro_f1']>best:
   best=metric['macro_f1'];stale=0;atomic_torch_save(folder/'best.pt',{'model':model.state_dict(),'metrics':metric,'epoch':epoch+1,'version':ARGS.version,'seed':seed,'role':role});np.savez_compressed(folder/'validation_predictions.npz',logits=logits)
  else:stale+=1
  atomic_torch_save(folder/'last.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state(),'next_epoch':epoch+1,'best':best,'stale':stale,'history':history});save(folder/'progress.json',history[-1]);print(ARGS.version,seed,role,epoch+1,'val',metric['macro_f1'],'best',best,flush=True)
  if stale>=patience:break
 save(folder/'result.json',{'version':ARGS.version,'seed':seed,'role':role,'best_val_f1':best,'epochs':len(history),'history':history,'test_read':False})

def preflight():
 d=load_data();seed_everything(42);models={k:Model(2,k,42).to(DEVICE) for k in ['evidence','rule','chain','full']}
 # Dedicated classifier RNG ensures new feature columns match across switches.
 assert all(torch.equal(models['evidence'].classifier[0].weight,m.classifier[0].weight) for m in models.values())
 a=d['train'];ix=np.arange(9);m=models['full'];m.train();loss,parts=m.loss(a['x'][ix],a['m'][ix],a['y'][ix],a['e'][ix]);loss.backward();assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
 m.zero_grad(set_to_none=True);r=F.cross_entropy(m.rule_head(m.evidence(a['e'][ix])),a['y'][ix]);grads=torch.autograd.grad(r,[m.gain_raw,*m.dvae.parameters()],allow_unused=True);assert grads[0].norm()>0 and all(x is None for x in grads[1:])
 save(OUT/'audit/preflight.json',{'status':'passed','classifier_input_dim':31,'adapted_stats24_retained':True,'score_residual_used':False,'shared_new_columns_match':True,'finite_backward':True,'R_updates_shared_gains_not_direct_encoder':True,'new7_prior':'empirical train-only class response, not valve priors','test_read':False})

def test():
 assert read(HOME/'all_training_complete.json')['complete']
 tf=read(OUT/'configs/transform.json');ep=OUT/'cache/test_e.npy'
 if not ep.exists():a,v=raw_features('test');np.save(ep,transform(a,v,tf))
 x=np.load(BASE/'cache/test/x.npy',mmap_mode='r');mask=np.load(BASE/'cache/test/mask.npy',mmap_mode='r');e=np.load(ep);idx=np.load(BASE/'cache/test/index.npy');rows=[]
 for seed in [42,2025,2026]:
  for role in ['stage1','evidence','rule','chain','full']:
   folder=OUT/'runs'/f'seed{seed}_{role}';dest=folder/'test_result.json'
   if dest.exists():rows.append(read(dest));continue
   if role=='stage1':
    source=BASE/('tuning' if ARGS.version=='1.1.0' else 'tuning_20261028')/'runs'/f'F_seed{seed}_baseline/test_result.json'
    if source.exists():
     old=read(source);cm=np.array(old['mixed']['confusion_matrix']);tp=np.diag(cm);p=np.divide(tp,cm.sum(0),out=np.zeros(9),where=cm.sum(0)>0);r=np.divide(tp,cm.sum(1),out=np.zeros(9),where=cm.sum(1)>0);f=np.divide(2*tp,cm.sum(0)+cm.sum(1),out=np.zeros(9),where=cm.sum(0)+cm.sum(1)>0);assert abs(f.mean()-old['mixed']['macro_f1'])<1e-12
     out={'seed':seed,'role':role,'macro_p':float(p.mean()),'macro_r':float(r.mean()),'macro_f1':float(f.mean()),'confusion_matrix':cm.tolist(),'selected_epoch':old['checkpoint_epoch'],'reused_existing_test':True,'source_test_report':str(source)};save(dest,out);rows.append(out);continue
   c=torch.load(folder/'best.pt',map_location=DEVICE);model=Model(1 if role=='stage1' else 2,role,seed).to(DEVICE).eval();model.load_state_dict(c['model']);outs=[]
   with torch.no_grad():
    for i in range(0,len(x),256):outs.append(model.predict(torch.tensor(np.array(x[i:i+256]),device=DEVICE),torch.tensor(np.array(mask[i:i+256]),device=DEVICE,dtype=torch.bool),torch.tensor(e[i:i+256],device=DEVICE)).cpu().numpy())
   logits=np.concatenate(outs);pred=logits.argmax(1);cm=np.bincount(idx[:,2]*9+pred,minlength=81).reshape(9,9);p=np.divide(np.diag(cm),cm.sum(0),out=np.zeros(9),where=cm.sum(0)>0);r=np.diag(cm)/cm.sum(1);f=np.divide(2*np.diag(cm),cm.sum(0)+cm.sum(1),out=np.zeros(9),where=(cm.sum(0)+cm.sum(1))>0);out={'seed':seed,'role':role,'macro_p':float(p.mean()),'macro_r':float(r.mean()),'macro_f1':float(f.mean()),'confusion_matrix':cm.tolist(),'selected_epoch':c['epoch']};np.savez_compressed(folder/'test_predictions.npz',logits=logits,predictions=pred,labels=idx[:,2],sources=idx[:,3]);save(dest,out);rows.append(out)
 save(OUT/'reports/test_rows.json',rows)
 for scope,roles in [('主表',['stage1','rule','chain','full']),('审计对照',['stage1','evidence','rule','chain','full'])]:
  lines=[f'# {ARGS.version} 适配P3基线与直接拼接新7D：{scope}','', '| 配置 | Macro-P±SD | Macro-R±SD | Macro-F1±SD |','|---|---:|---:|---:|']
  for role in roles:
   rs=[x for x in rows if x['role']==role];vals=[f'{statistics.mean(x[k] for x in rs):.6f} ± {statistics.stdev(x[k] for x in rs):.6f}' for k in ['macro_p','macro_r','macro_f1']];lines.append('| '+role+' | '+' | '.join(vals)+' |')
  (OUT/'reports'/f'{scope}_PRF.md').write_text('\n'.join(lines),encoding='utf-8')

def reuse_baselines():
 source=BASE/('tuning' if ARGS.version=='1.1.0' else 'tuning_20261028')
 for seed in [42,2025,2026]:
  src=source/'runs'/f'F_seed{seed}_baseline';dst=OUT/'runs'/f'seed{seed}_stage1';dst.mkdir(exist_ok=True)
  checkpoint=torch.load(src/'best.pt',map_location='cpu');assert checkpoint['model']['classifier.0.weight'].shape==(64,24)
  if (dst/'best.pt').exists():assert sha(dst/'best.pt')==sha(src/'best.pt')
  else:shutil.copy2(src/'best.pt',dst/'best.pt')
  report=read(src/'result.json');save(dst/'result.json',{'reused':True,'source':str(src),'checkpoint_sha256':sha(src/'best.pt'),'best_val_f1':checkpoint['metrics']['macro_f1'],'version':ARGS.version,'seed':seed,'role':'stage1','epochs':0,'original_report':report})

def audit_baselines():
 source=BASE/('tuning' if ARGS.version=='1.1.0' else 'tuning_20261028');sys.path.insert(0,str(source/'src'))
 from tuned_model import TunedDiagnoser
 d=load_data();results=[]
 for seed in [42,2025,2026]:
  checkpoint=torch.load(OUT/'runs'/f'seed{seed}_stage1/best.pt',map_location=DEVICE);state=checkpoint['model'];m=Model(1,'stage1',seed).to(DEVICE).eval();own=m.state_dict()
  assert all(k in own and own[k].shape==v.shape for k,v in state.items());own.update(state);m.load_state_dict(own)
  original=TunedDiagnoser(read(BASE/'configs/normalization.json')['features'],checkpoint['config']).to(DEVICE).eval();original.load_state_dict(state)
  a=d['val'];ix=np.arange(min(128,len(a['y'])));x,mask,y,e=[a[k][ix] for k in ['x','m','y','e']]
  with torch.no_grad():
   first=m.predict(x,mask,e);second=original.predict(x,mask);assert torch.equal(first,second)
   torch.manual_seed(7788);new_loss=m.loss(x,mask,y,e)[0];torch.manual_seed(7788);old_loss=original.loss(x,mask,y)[0];assert torch.allclose(new_loss,old_loss,atol=1e-6,rtol=1e-6),(new_loss,old_loss)
  metric,_=evaluate(m,d['val']);assert abs(metric['macro_f1']-checkpoint['metrics']['macro_f1'])<1e-12
  results.append({'seed':seed,'baseline_weights_exact':True,'logits_max_difference':float((first-second).abs().max()),'primary_loss_matches_adapted':True,'validation_f1':metric['macro_f1'],'validation_matches_original':True})
  del m,original
 save(OUT/'audit/adapted_baseline_identity.json',{'status':'passed','results':results,'test_read':False})


def main():
 for d in ['configs','cache','runs','reports','audit']:(OUT/d).mkdir(parents=True,exist_ok=True)
 if ARGS.role=='prepare':prepare();reuse_baselines();audit_baselines()
 elif ARGS.role=='preflight':preflight()
 elif ARGS.role=='test':test()
 else:train()
if __name__=='__main__':main()
