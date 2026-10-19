"""xBD data adapter. External label 0=damage; PUDA internal label 1=damage."""
import ast
import hashlib
import csv
import importlib.util
import json
from pathlib import Path
import random
import sys
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
DATA = Path('/home/qiruiwang/data2/xbd_building_crops_by_disaster/xBD')
PRIORS = {'joplin-tornado': .4379, 'lower-puna-volcano': .3591, 'santa-rosa-wildfire': .7886}
U_ROOTS = {'joplin-tornado': '/home/qiruiwang/joplin_tornado_building_patches_64',
           'lower-puna-volcano': '/home/qiruiwang/lower-puna-volcanou',
           'santa-rosa-wildfire': '/home/qiruiwang/santa-rosa-wildfire_building_patches_64'}
EXPERIMENT = ROOT / 'source_adapter'
LABELS = {'destroyed': 0, 'major-damage': 0, 'minor-damage': 1, 'no-damage': 1}
EXTENSIONS = {'.jpg', '.jpeg', '.png'}

def transform():
    return transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop((221, 221)),
        transforms.ToTensor(), transforms.Normalize([.485,.456,.406], [.229,.224,.225])])

def scan(root):
    if not root.is_dir():
        raise FileNotFoundError(root)
    return [(str(p), LABELS[d.name]) for d in sorted(root.iterdir()) if d.is_dir() and d.name in LABELS
            for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in EXTENSIONS]

class Images(Dataset):
    # items: path, PU label (1/-1), true internal label (1/0/-1 unknown)
    def __init__(self, items):
        self.items = items
        self.transform = transform()
    def __len__(self):
        return len(self.items)
    def load_image(self, path):
        with Image.open(path) as img:
            return self.transform(img.convert('RGB'))
    def __getitem__(self, index):
        path, pu, truth = self.items[index]
        return self.load_image(path), pu, truth, index

class PseudoDataset(Images):
    def __getitem__(self, index):
        path, prob = self.items[index]
        return self.load_image(path), prob

def build_data(args):
    # Use the actual source training collection/split functions, including suffix exclusions.
    spec = importlib.util.spec_from_file_location('puda_source_split', EXPERIMENT / 'train_source_remoteclip_rn50.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    all_source, excluded, ignored = module.collect_samples(args.source_root)
    source, source_val = module.split_samples(all_source, .2, args.seed)
    args.p_root = args.p_root or DATA / (args.dataset + '-train')
    args.u_root = args.u_root or Path(U_ROOTS[args.dataset])
    args.test_root = args.test_root or DATA / (args.dataset + '-test')
    pool = scan(args.p_root)
    random.Random(args.seed).shuffle(pool)
    positives = [(p, label) for p,label in pool if label == 0][:args.p_count]
    if len(positives) != args.p_count:
        raise ValueError(f'Need {args.p_count} P, found {len(positives)}')
    if not args.u_root.is_dir():
        raise FileNotFoundError(args.u_root)
    unlabeled = [(str(p), 1) for p in sorted(args.u_root.iterdir()) if p.is_file() and p.suffix.lower() in EXTENSIONS]
    random.Random(args.seed).shuffle(unlabeled)
    test = scan(args.test_root)
    if not unlabeled or not test:
        raise ValueError('Empty U or test')
    if {p for p,_ in positives} & {p for p,_ in test}:
        raise ValueError('P/test path overlap')
    # Original target loader includes both labeled P and U; additional P is appended per step.
    source_set = Images([(s.path, 1 if s.label == 0 else -1, 1-s.label) for s in source])
    target_set = Images([(p,1,1) for p,_ in positives] + [(p,-1,-1) for p,_ in unlabeled])
    test_set = Images([(p,-1,1-y) for p,y in test])
    audit = dict(dataset=args.dataset, seed=args.seed, source_count=len(source), source_val_unused=len(source_val),
                 excluded_families=list(module.EXCLUDED), excluded_directories=excluded, ignored_directories=ignored,
                 p_count=len(positives), u_count=len(unlabeled), test_count=len(test), prior=args.prior,
                 prior_origin='User supplied; interpreted as damage fraction for nnPU; not estimated from test',
                 internal_positive=1, external_damage_label=0,
                 p_batch=args.p_batch, target_loader='P + all U; additional sampled P each step')
    return source_set, target_set, test_set, positives, audit

def write_audit(args, audit, source, target, test):
    out=args.output_dir
    (out/'config.json').write_text(json.dumps(vars(args),default=str,indent=2)+'\n')
    (out/'data_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    for name, ds in [('source',source),('target',target),('test',test)]:
        with (out/f'{name}_samples.jsonl').open('w') as f:
            for row in ds.items:
                f.write(json.dumps(row)+'\n')
    (out/'p_samples.json').write_text(json.dumps([[p,0] for p,pu,y in target.items if pu==1],indent=2))

def load_imagenet(path=None):
    # Reuse upstream key mapping without its import-time download/save side effects.
    tree=ast.parse((ROOT/'pretrained_model_rename.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='key_transformation')
    scope={}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'upstream_key_mapping','exec'),scope)
    if path:
        try:
            state=torch.load(path,map_location='cpu',weights_only=True)
        except RuntimeError as error:
            if 'legacy .tar format' not in str(error):
                raise
            # Original official torchvision checkpoint uses legacy tar serialization.
            expected='19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097'
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
                raise RuntimeError('Refusing legacy unpickling for an unverified checkpoint') from error
            state=torch.load(path,map_location='cpu',weights_only=False)
    else:
        state=torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth',map_location='cpu',check_hash=True)
    return {scope['key_transformation'](k): v for k,v in state.items() if scope['key_transformation'](k) is not None}

@torch.no_grad()
def evaluate_test(g,f,ds,batch,path):
    modes=g.training,f.training
    rng=torch.get_rng_state()
    g.eval();f.eval()
    cm=torch.zeros(2,2,dtype=torch.long)
    device=next(g.parameters()).device
    predictions=[]
    for x,_,truth,indices in DataLoader(ds,batch_size=batch,shuffle=False,num_workers=6):
        logits=f(g(x.to(device)))[1]
        pred=1-logits.argmax(1).cpu()  # back to damage=0
        truth=1-truth
        cm+=torch.bincount(truth*2+pred,minlength=4).reshape(2,2)
        predictions.extend((ds.items[i][0],int(y),int(p)) for i,y,p in zip(indices.tolist(),truth,pred))
    tp,fn,fp,tn=cm.flatten().tolist()
    result=dict(oa=(tp+tn)/int(cm.sum()),damage_f1=2*tp/max(1,2*tp+fp+fn),positive_label=0,
                confusion_matrix=cm.tolist(),samples=int(cm.sum()))
    path.write_text(json.dumps(result,indent=2)+'\n')
    with path.with_suffix('.csv').open('w',newline='') as handle:
        w=csv.writer(handle);w.writerow(['path','true_label','pred_label']);w.writerows(predictions)
    g.train(modes[0]);f.train(modes[1]);torch.set_rng_state(rng)
    return result

def save_stage(path,g,f,args,stage,epoch):
    torch.save(dict(encoder=g.state_dict(),classifier=f.state_dict(),stage=stage,epoch=epoch,
                    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},positive_label_internal=1),path)
