#!/usr/bin/env python3
"""PUDA xBD baseline. See XBD_NOTES.txt for upstream quirks retained verbatim."""
import argparse
import json
import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.net import ResNet50_mod_name, ResClassifier
from pu_loss import PULoss
from utils import weights_init
from xbd_data import (DATA, PRIORS, build_data, write_audit, load_imagenet,
                      evaluate_test, save_stage, PseudoDataset)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=list(PRIORS),required=True)
    p.add_argument('--seed',type=int,choices=[922,923,924],required=True)
    p.add_argument('--source-root',type=Path,default=DATA)
    for name in ['p-root','u-root','test-root','imagenet-weights']:
        p.add_argument('--'+name,type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--prior',type=float)
    p.add_argument('--p-count',type=int,default=400)
    p.add_argument('--p-batch',type=int,default=16)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--stage2-epochs',type=int,default=50)
    p.add_argument('--lr1',type=float,default=.0016102287018430657)
    p.add_argument('--lr2',type=float,default=.00012266834924870657)
    p.add_argument('--check-data',action='store_true')
    p.add_argument('--stage1-only',action='store_true',help='Stop after saving stage1_last.pt; stage2 can run separately')
    p.add_argument('--check-epoch-boundary',action='store_true',help='Diagnostic only: two short epochs retaining the real final batch size')
    p.add_argument('--smoke-test',action='store_true',help='Two batches per stage on small subsets; metrics are NOT experiment results')
    a=p.parse_args()
    if a.prior is None: a.prior=PRIORS[a.dataset]
    if not 0<a.prior<1 or not 1<a.p_batch<=min(a.p_count,a.batch_size) or min(a.epochs,a.stage2_epochs)<1:
        p.error('Require 0<prior<1, 2<=p_batch<=min(p_count,batch_size), positive epochs')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    source,target,test,positives,audit=build_data(a)
    write_audit(a,audit,source,target,test)
    print(json.dumps(audit),flush=True)
    if a.check_data:return
    if not torch.cuda.is_available():raise RuntimeError('Upstream PUDA decoder requires CUDA')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    # MaxUnpool2d in the upstream decoder has no deterministic CUDA implementation.
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    initial=load_imagenet(a.imagenet_weights)
    def new_models():
        g=ResNet50_mod_name().cuda();g.load_state_dict(initial,strict=True)
        f=ResClassifier(class_num=2,dropout_p=.1).cuda();f.apply(weights_init)
        return g,f
    if a.smoke_test:
        a.batch_size=a.p_batch=2
        source.items=[r for y in (0,1) for r in [s for s in source.items if s[2]==y][:4]]
        target.items=target.items[:2]+[r for r in target.items if r[1]==-1][:6]
        test.items=[r for y in (0,1) for r in [s for s in test.items if s[2]==y][:2]]
        positives=positives[:2]
    source_loader=DataLoader(source,batch_size=a.batch_size,shuffle=True,num_workers=6)
    target_loader=DataLoader(target,batch_size=a.batch_size,shuffle=True,num_workers=6)
    all_p=torch.stack([target.load_image(path) for path,_ in positives])
    sampling=torch.Generator().manual_seed(a.seed+10000)
    g,f=new_models()
    if a.check_epoch_boundary:
        if not a.stage1_only or a.smoke_test:
            raise ValueError('Boundary diagnostic requires --stage1-only and no --smoke-test')
        remainder=len(target)%a.batch_size
        if remainder == 0: raise ValueError('No incomplete batch to reproduce')
        target.items=target.items[:2*a.batch_size+remainder]
        a.epochs=2
        print('DIAGNOSTIC ONLY: shortened epochs; metrics are not experiment results',flush=True)
    og=torch.optim.SGD(g.parameters(),lr=a.lr1,weight_decay=.0005)
    of=torch.optim.SGD(f.parameters(),lr=a.lr1,momentum=.9,weight_decay=.0005)
    pu=PULoss(prior=torch.tensor(a.prior,dtype=torch.float32),nnPU=True)
    n=a.batch_size+a.p_batch
    recon_t=torch.ones(n,3,221,221,device='cuda');recon_z=torch.ones_like(recon_t)
    pos_source=[.5];neg_source=[.5];target_thresholds=[];history=[]
    log=(a.output_dir/'history.jsonl').open('w')
    best = {'oa': -1.}
    def record_test(encoder, classifier, stage, epoch):
        folder=a.output_dir/'epoch_tests';folder.mkdir(exist_ok=True)
        m=evaluate_test(encoder,classifier,test,a.batch_size,folder/f'stage{stage}_epoch{epoch:03d}.json')
        m.update(stage=stage,epoch=epoch)
        (folder/f'stage{stage}_epoch{epoch:03d}.json').write_text(json.dumps(m,indent=2))
        for key,name in [('oa','best_oa')]:
            if m[key]>best[key]:
                best[key]=m[key]
                save_stage(a.output_dir/f'{name}.pt',encoder,classifier,a,f'stage{stage}',epoch)
                (a.output_dir/f'{name}.json').write_text(json.dumps(m,indent=2))
        save_stage(a.output_dir/'last.pt',encoder,classifier,a,f'stage{stage}',epoch)
        (a.output_dir/'last.json').write_text(json.dumps(m,indent=2))
        return m
    steps=0
    for epoch in range(1,(1 if a.smoke_test else a.epochs)+1):
        # All P determine the mean; chunked train-mode BN is a documented memory adaptation.
        with torch.no_grad():
            chunks = torch.tensor_split(all_p, max(1, (len(all_p) + a.batch_size - 1) // a.batch_size))
            threshold = sum(float(F.softmax(f(g(chunk.cuda()))[1],dim=1)[:,1].sum()) for chunk in chunks) / len(all_p)
        target_thresholds.append(threshold)
        prev_pos=float(np.mean(pos_source));prev_neg=float(np.mean(neg_source))
        pos_source=[];neg_source=[]
        sit=iter(source_loader);loss_sum=0.;updates=0
        for batch,(tx,tpu,_,indices) in enumerate(tqdm(target_loader,desc=f'PUDA stage1 {epoch}')):
            try:sx,_,sy,_=next(sit)
            except StopIteration:
                sit=iter(source_loader);sx,_,sy,_=next(sit)
            # Authorized change: sample P from the fixed 400-image pool each step.
            pb=all_p[torch.randperm(len(all_p),generator=sampling)[:a.p_batch]]
            tx=torch.cat([tx,pb]);tpu=torch.cat([tpu,torch.ones(a.p_batch,dtype=torch.long)])
            indices=torch.cat([indices,torch.full((a.p_batch,),-1)])
            sx=torch.cat([sx,sx[:a.p_batch]]);sy=torch.cat([sy,sy[:a.p_batch]])
            # Preserve upstream final-incomplete-batch behavior, without diagnostics on unknown labels.
            if len(sx)<=1 or len(tx)<=1:continue
            sx,sy,tx,tpu=sx.cuda(),sy.cuda(),tx.cuda(),tpu.cuda()
            z=torch.tensor(np.random.normal(0,1,(n,2048)),device='cuda',dtype=torch.float32)
            og.zero_grad();of.zero_grad()
            sf=g(sx);tf=g(tx);se,sl=f(sf);te,tl=f(tf)
            pos_source.append(float(F.softmax(sl[sy==1],dim=1)[:,1].mean().detach()))
            neg_source.append(float(F.softmax(sl[sy==0],dim=1)[:,1].mean().detach()))
            prob=F.softmax(tl[tpu==-1],dim=1)[:,1]
            for mask in (prob>(prev_pos+threshold)/2,prob<prev_neg):
                if epoch>20 or a.smoke_test:
                    history.extend(zip(indices[:len(indices)-a.p_batch][tpu[:-a.p_batch].cpu()==-1][mask.cpu()].tolist(),prob[mask].detach().cpu().tolist()))
            if len(sx)!=n or len(tx)!=n:continue
            def norm_loss(x):
                return .05*((x.norm(p=2,dim=1)-(x.norm(p=2,dim=1).detach()+1))**2).mean()
            tp=F.softmax(tl,dim=1);v=tp[tp>=.000001]
            entropy=-.05*(v*v.log()).sum()/len(tp)
            kl=F.kl_div(F.log_softmax(sf.reshape(-1,2048),dim=1),F.softmax(z,dim=1),reduction='mean')
            dal=F.l1_loss(recon_t,recon_z)  # Upstream detached previous-step reconstruction, intentionally retained.
            loss=.93*F.nll_loss(F.log_softmax(sl,dim=1),sy)+.86*(norm_loss(se)+norm_loss(te)+entropy+.049*kl+.2*dal)+pu(tl[:,1],tpu)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite PUDA stage1 loss')
            loss.backward();og.step();of.step()
            with torch.no_grad():
                recon_t=g(tx,is_deconv=True).detach();recon_z=g.decode(z).detach()
            loss_sum+=float(loss.detach());updates+=1;steps+=1
            if a.smoke_test and updates>=2:break
        if not updates:raise RuntimeError('No optimizer updates in stage1')
        row=dict(stage=1,epoch=epoch,loss=loss_sum/updates,updates=updates,target_threshold=threshold)
        row['cuda_allocated_gib']=torch.cuda.memory_allocated()/2**30
        row['cuda_peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
        row['test']=record_test(g,f,1,epoch)
        print(json.dumps(row),flush=True);log.write(json.dumps(row)+'\n');log.flush()
    save_stage(a.output_dir/'stage1_last.pt',g,f,a,'stage1',epoch)
    metrics=evaluate_test(g,f,test,a.batch_size,a.output_dir/'stage1_test.json')
    if a.stage1_only:
        log.close()
        print('Stage1 complete: '+str(a.output_dir/'stage1_last.pt'),flush=True)
        return
    frame=pd.DataFrame(history,columns=['image','soft_label'])
    pt=float(np.mean(target_thresholds));nt=1-pt
    pseudo=frame.groupby('image')['soft_label'].apply(lambda x:-1 if len(x)<10 else x.mean() if (x.max()<nt or x.min()>pt) else -1)
    items=[(target.items[int(i)][0],float(v)) for i,v in pseudo.items() if v!=-1]
    if a.smoke_test:
        # Synthetic pseudo-labels exercise stage2 only; never used in real experiments.
        items=[(path,.8) for path,pu_label,_ in target.items if pu_label==-1][:2]
    if items:
        items += [(path,1.) for path,_ in positives]
        (a.output_dir/'pseudo_samples.json').write_text(json.dumps(items,indent=2))
        del g,f,og,of,recon_t,recon_z
        g2,f2=new_models()
        og2=torch.optim.SGD(g2.parameters(),lr=a.lr2,weight_decay=.0005)
        of2=torch.optim.SGD(f2.parameters(),lr=a.lr2,momentum=.9,weight_decay=.0005)
        sg=torch.optim.lr_scheduler.StepLR(og2,35,.0035);ss=torch.optim.lr_scheduler.StepLR(of2,35,.0035)
        loader=DataLoader(PseudoDataset(items),batch_size=a.batch_size,shuffle=True,num_workers=6)
        print('Upstream stage2 behavior retained: no zero_grad between batches; eval mode after first update; scheduler steps every batch.',flush=True)
        for epoch2 in range(1,(1 if a.smoke_test else a.stage2_epochs)+1):
            losses=[]
            for x,y in tqdm(loader,desc=f'PUDA stage2 {epoch2}'):
                x=x.cuda();y=(y>.5).long().cuda()
                loss=F.cross_entropy(f2(g2(x))[1],y)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite PUDA stage2 loss')
                # Exact upstream behavior (see notes); no unapproved optimization fixes.
                loss.backward();og2.step();of2.step();sg.step();ss.step()
                g2.eval();f2.eval()
                losses.append(float(loss.detach()))
                if a.smoke_test and len(losses)>=2:break
            row=dict(stage=2,epoch=epoch2,loss=float(np.mean(losses)),updates=len(losses))
            row['test']=record_test(g2,f2,2,epoch2)
            log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
        save_stage(a.output_dir/'stage2_last.pt',g2,f2,a,'stage2',epoch2)
        metrics=evaluate_test(g2,f2,test,a.batch_size,a.output_dir/'final_test.json')
        metrics['stage2_status']='completed'
    else:
        metrics['stage2_status']='no_selected_pseudo_labels; stage1 fallback'
    metrics['smoke_test']=a.smoke_test
    (a.output_dir/'final_test.json').write_text(json.dumps(metrics,indent=2)+'\n')
    log.close();print(json.dumps(metrics),flush=True)

if __name__=='__main__':main()
