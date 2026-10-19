#!/usr/bin/env python3
"""PUDA variant: fresh last-checkpoint pseudo labels and conventional stage2 CE."""
import argparse
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.net import ResNet50_mod_name, ResClassifier
from utils import weights_init
from xbd_data import DATA, Images, PseudoDataset, EXTENSIONS, scan, load_imagenet, evaluate_test, save_stage

ROOT = Path(__file__).resolve().parent

def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str) + '\n')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'stage1_last.pt')
    p.add_argument('--original-pseudo', type=Path, default=ROOT/'pseudo_samples.json')
    p.add_argument('--p-manifest', type=Path, help='Exact stage1 p_samples.json; preferred for new runs')
    p.add_argument('--u-root', type=Path, default=Path('/home/qiruiwang/tornadou'))
    p.add_argument('--p-root', type=Path, default=DATA/'joplin-tornado-train')
    p.add_argument('--test-root', type=Path, default=DATA/'joplin-tornado-test')
    p.add_argument('--imagenet-weights', type=Path, default=Path('/home/qiruiwang/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth'))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr2', type=float, default=.00012266834924870657)
    a = p.parse_args()
    if a.batch_size < 2 or a.epochs < 1: p.error('batch-size >= 2 and epochs >= 1 required')
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    if ck['positive_label_internal'] != 1 or str(ck['stage']) not in ('stage1', '1'):
        raise ValueError('Expected stage1 checkpoint with internal positive label 1')
    a.seed = int(ck['config']['seed'])
    a.dataset = ck['config']['dataset']
    if a.p_root.name != a.dataset+'-train' or a.test_root.name != a.dataset+'-test':
        raise ValueError('P/test roots must match the checkpoint dataset')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    out = a.output_dir
    write(out/'config.json', vars(a))
    write(out/'status.json', {'state': 'generating_pseudo_labels'})
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)
    def seed():
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    seed()
    # Recover exact original P identities, not a new random selection.
    old = json.loads((a.p_manifest or a.original_pseudo).read_text())
    positives = [str(a.p_root/Path(path).parent.name/Path(path).name) for path, prob in old
                 if Path(path).parent.name in ('destroyed', 'major-damage')]
    if len(positives) != 400 or len(set(positives)) != 400 or not all(Path(x).is_file() for x in positives):
        raise ValueError('Could not recover exact 400 original positives')
    pool = scan(a.p_root); random.Random(a.seed).shuffle(pool)
    expected = [path for path, y in pool if y == 0][:400]
    if set(expected) != set(positives): raise ValueError('Original P does not match checkpoint seed')
    paths = [str(x) for x in sorted(a.u_root.iterdir()) if x.is_file() and x.suffix.lower() in EXTENSIONS]
    if not paths: raise ValueError('Empty U')
    test = Images([(path, -1, 1-y) for path, y in scan(a.test_root)])
    if not len(test): raise ValueError('Empty test')
    write(out/'p_samples.json', [[x, 0] for x in positives])
    g = ResNet50_mod_name().cuda(); f = ResClassifier(class_num=2, dropout_p=.1).cuda()
    g.load_state_dict(ck['encoder'], strict=True); f.load_state_dict(ck['classifier'], strict=True)
    del ck
    baseline = evaluate_test(g, f, test, a.batch_size, out/'stage1_test.json')
    print('Stage1 last test: '+json.dumps(baseline), flush=True)
    g.eval(); f.eval()
    items = []
    ds = Images([(x, -1, -1) for x in paths])
    with torch.no_grad():
        for x, _, _, indices in tqdm(DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.workers), desc='Fresh U pseudo labels'):
            probs = f(g(x.cuda()))[1].softmax(1)[:, 1].cpu().tolist()
            items.extend((paths[i], prob) for i, prob in zip(indices.tolist(), probs))
    npos = sum(prob > .5 for _, prob in items)
    items.extend((x, 1.) for x in positives)
    write(out/'pseudo_samples.json', items)
    audit = dict(seed=a.seed, u_count=len(paths), u_damage=npos, u_non_damage=len(paths)-npos,
                 p_count=400, train_count=len(items), test_count=len(test), threshold=.5,
                 internal_positive=1, external_damage_label=0, history_accumulation=False,
                 pseudo_mode='eval/no_grad; final stage1 checkpoint; all U',
                 stage2_initialization='ImageNet encoder + randomly initialized original classifier',
                 scheduler='StepLR step_size=35 gamma=0.0035 once per epoch', precision='FP32')
    write(out/'data_audit.json', audit); print(json.dumps(audit), flush=True)
    del g, f; torch.cuda.empty_cache()
    seed()
    g = ResNet50_mod_name().cuda(); g.load_state_dict(load_imagenet(a.imagenet_weights), strict=True)
    f = ResClassifier(class_num=2, dropout_p=.1).cuda(); f.apply(weights_init)
    og = torch.optim.SGD(g.parameters(), lr=a.lr2, weight_decay=.0005)
    of = torch.optim.SGD(f.parameters(), lr=a.lr2, momentum=.9, weight_decay=.0005)
    schedulers = [torch.optim.lr_scheduler.StepLR(opt, step_size=35, gamma=.0035) for opt in (og, of)]
    if len(items) % a.batch_size == 1: raise ValueError('Choose batch size avoiding a singleton BatchNorm batch')
    loader = DataLoader(PseudoDataset(items), batch_size=a.batch_size, shuffle=True,
                        num_workers=a.workers, persistent_workers=a.workers > 0)
    (out/'epoch_tests').mkdir()
    best = -1.
    for epoch in range(1, a.epochs+1):
        g.train(); f.train()
        used_lr = [opt.param_groups[0]['lr'] for opt in (og, of)]
        total_loss = 0.; count = 0; updates = 0
        write(out/'status.json', {'state':'training', 'epoch':epoch, 'epochs':a.epochs})
        for x, prob in tqdm(loader, desc=f'Corrected stage2 {epoch}'):
            x = x.cuda(); y = (prob > .5).long().cuda()
            og.zero_grad(set_to_none=True); of.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(f(g(x))[1], y)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward(); og.step(); of.step()
            total_loss += float(loss.detach())*len(x); count += len(x); updates += 1
        for scheduler in schedulers: scheduler.step()
        metrics = evaluate_test(g, f, test, a.batch_size, out/'epoch_tests'/f'stage2_epoch{epoch:03d}.json')
        metrics.update(stage=2, epoch=epoch)
        row = dict(epoch=epoch, lr=used_lr, loss=total_loss/count, updates=updates, samples=count, test=metrics)
        with (out/'history.jsonl').open('a') as log: log.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
        save_stage(out/'last.pt', g, f, a, 'stage2', epoch); write(out/'last.json', metrics)
        if metrics['oa'] > best:
            best = metrics['oa']; save_stage(out/'best_oa.pt', g, f, a, 'stage2', epoch); write(out/'best_oa.json', metrics)
        write(out/'status.json', {'state':'training', 'completed_epochs':epoch, 'last':metrics})
    write(out/'status.json', {'state':'completed', 'last':metrics, 'best_oa':best})
    print('Finished: '+str(out), flush=True)

if __name__ == '__main__':
    main()
