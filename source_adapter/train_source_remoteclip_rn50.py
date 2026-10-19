#!/usr/bin/env python3
"""RemoteCLIP RN50 source-only binary training for PANDA.

Independent of trainsource.py. Both source-only evaluation and PANDA must load
the SAME best_source.pt and use the saved preprocessing/feature convention.
Example: python train_source_remoteclip_rn50.py --batch-size 64
Data audit only: python train_source_remoteclip_rn50.py --check-data
Two training-step check: python train_source_remoteclip_rn50.py --smoke-test
"""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import random

import open_clip
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


BASE = Path(__file__).resolve().parent
DATA_ROOT = Path('/home/qiruiwang/data2/xbd_building_crops_by_disaster/xBD')
EXCLUDED = (
    'joplin-tornado', 'lower-puna-volcano', 'moore-tornado',
    'santa-rosa-wildfire', 'tuscaloosa-tornado',
)
CLASS_TO_LABEL = {
    'destroyed': 0, 'major-damage': 0, 'minor-damage': 1, 'no-damage': 1,
}
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


@dataclass(frozen=True)
class Sample:
    path: str
    label: int
    disaster: str
    damage_class: str


def is_excluded(name):
    # Includes canonical folders and derived folders such as -train/-test.
    name = name.lower().replace('_', '-')
    return any(name == base or name.startswith(base + '-') for base in EXCLUDED)


def collect_samples(root):
    samples, excluded_dirs, ignored_dirs = [], [], []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        if is_excluded(folder.name):
            excluded_dirs.append(folder.name)
            continue
        before = len(samples)
        for damage_class, label in CLASS_TO_LABEL.items():
            class_dir = folder / damage_class
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in EXTENSIONS:
                    samples.append(Sample(str(path), label, folder.name, damage_class))
        if len(samples) == before:
            ignored_dirs.append(folder.name)
    if not samples:
        raise ValueError(f'No source images found under {root}')
    return samples, excluded_dirs, ignored_dirs


def split_samples(samples, val_ratio, seed):
    # Match trainsource.py: stratify by binary label, using the same RNG order.
    by_label = {}
    for sample in samples:
        by_label.setdefault(sample.label, []).append(sample)
    if set(by_label) != {0, 1}:
        raise ValueError('Both binary classes are required.')
    rng = random.Random(seed)
    train, val = [], []
    for group in by_label.values():
        group = list(group)
        rng.shuffle(group)
        boundary = int(len(group) * (1 - val_ratio))
        if not 0 < boundary < len(group):
            raise ValueError('Each class needs samples in both train and validation.')
        train.extend(group[:boundary])
        val.extend(group[boundary:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class XBDDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        try:
            with Image.open(sample.path) as image:
                image = self.transform(image.convert('RGB'))
        except Exception as error:
            raise RuntimeError(f'Failed to read image: {sample.path}') from error
        return image, sample.label


class RemoteCLIPClassifier(nn.Module):
    def __init__(self, visual, feature_dim):
        super().__init__()
        self.visual = visual
        self.classifier = nn.Linear(feature_dim, 2)

    def forward(self, images):
        # RN50's native attention-pool output: 1024 dimensions, no L2 normalize.
        return self.classifier(self.visual(images))


def build_model(checkpoint):
    clip = open_clip.create_model('RN50', pretrained=None)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    print('RemoteCLIP weights:', clip.load_state_dict(state, strict=True), flush=True)
    preprocess = {
        'image_size': 224, 'resize_mode': 'squash', 'interpolation': 'bicubic',
        'normalize_mean': list(clip.visual.preprocess_cfg['mean']),
        'normalize_std': list(clip.visual.preprocess_cfg['std']),
        'color_mode': 'RGB', 'augmentation': False,
    }
    feature_dim = clip.visual.output_dim
    if feature_dim != 1024:
        raise ValueError(f'Unexpected RN50 output dimension: {feature_dim}')
    model = RemoteCLIPClassifier(clip.visual, feature_dim)
    # Text encoder goes out of scope here; only visual + classifier are trained.
    return model, preprocess


def make_transform(preprocess):
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(preprocess['normalize_mean'], preprocess['normalize_std']),
    ])


def optimizer_step_with_amp(model, optimizer, scaler, max_norm=2.):
    """Unscale first; let GradScaler skip non-finite gradients before clipping.

    Returns (updated, old_scale, new_scale). FP32 failures still raise.
    """
    old_scale = scaler.get_scale()
    scaler.unscale_(optimizer)
    parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not parameters:
        raise RuntimeError('No gradients found for the training parameters.')
    try:
        nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    except RuntimeError:
        # With error_if_nonfinite=True, clipping has not modified the gradients.
        finite = torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all()
        if not finite:
            if not scaler.is_enabled():
                raise RuntimeError('Non-finite gradients in FP32; inspect inputs and model.')
            # unscale_ recorded the overflow. step skips the optimizer update,
            # and update lowers the scale. Never clip these Inf/NaN gradients.
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            return False, old_scale, scaler.get_scale()
        # Finite elements can still overflow an FP32 norm reduction. Calculate
        # the norm in FP64 and multiply by the coefficient before casting it.
        with torch.no_grad():
            norm = torch.linalg.vector_norm(torch.stack([
                torch.linalg.vector_norm(p.grad, dtype=torch.float64)
                for p in parameters
            ]))
            if not torch.isfinite(norm):
                raise RuntimeError('Gradient norm remains non-finite in FP64.')
            coefficient = (max_norm / (norm + 1e-6)).clamp(max=1.)
            for p in parameters:
                p.grad.copy_((p.grad.double() * coefficient).to(p.grad.dtype))
    scaler.step(optimizer)
    scaler.update()
    return True, old_scale, scaler.get_scale()


def make_stage_scaler(stage, enabled, device_type='cuda'):
    # LP only differentiates the classifier. FT differentiates the whole visual
    # encoder and must not inherit LP's much larger calibrated loss scale.
    return torch.amp.GradScaler(
        device_type, enabled=enabled,
        init_scale=128. if stage == 'lp' else 1.,
        growth_interval=2000,
    )


def run_epoch(model, loader, device, amp, stage, optimizer=None, scaler=None,
              max_batches=None):
    training = optimizer is not None
    model.train(training)
    if training and stage == 'lp':
        # Freeze BatchNorm running statistics as well as visual parameters.
        model.visual.eval()
    confusion = torch.zeros(2, 2, dtype=torch.long, device=device)
    total_loss, count = 0., 0
    optimizer_steps, amp_skipped_steps, consecutive_skips = 0, 0, 0
    with torch.set_grad_enabled(training):
        for batch_index, (images, labels) in enumerate(tqdm(
            loader, desc=f'{stage} {"train" if training else "source val"}',
            total=min(len(loader), max_batches) if max_batches else len(loader),
        )):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss encountered.')
            if training:
                scaler.scale(loss).backward()
                updated, old_scale, new_scale = optimizer_step_with_amp(model, optimizer, scaler)
                if updated:
                    optimizer_steps += 1
                    consecutive_skips = 0
                else:
                    amp_skipped_steps += 1
                    consecutive_skips += 1
                    tqdm.write(
                        f'WARNING: {stage} batch {batch_index + 1}: AMP gradient overflow; '
                        f'optimizer update skipped, scale {old_scale:g} -> {new_scale:g}; '
                        f'skipped this epoch={amp_skipped_steps}'
                    )
                    if consecutive_skips >= 10 or amp_skipped_steps >= 100:
                        raise RuntimeError(
                            'Repeated AMP overflow (10 consecutive or 100 in one epoch). '
                            'Stop and inspect numerical stability; try --no-amp.'
                        )
            total_loss += loss.detach().item() * labels.size(0)
            count += labels.size(0)
            confusion += torch.bincount(
                labels * 2 + logits.detach().argmax(1), minlength=4,
            ).reshape(2, 2)
            if max_batches and batch_index + 1 >= max_batches:
                break
    if training and optimizer_steps == 0:
        raise RuntimeError('No successful optimizer updates in this epoch.')
    matrix = confusion.cpu().double()
    f1 = 2 * matrix.diag() / (matrix.sum(0) + matrix.sum(1)).clamp_min(1)
    return {
        'loss': total_loss / count,
        'accuracy': matrix.diag().sum().item() / count,
        'macro_f1': f1.mean().item(), 'damage_f1': f1[0].item(),
        'non_damage_f1': f1[1].item(), 'confusion_matrix': matrix.long().tolist(),
        # Training metrics include finite forward passes whose AMP update was
        # skipped; record the update counts separately for an accurate audit.
        'samples': count, 'optimizer_steps': optimizer_steps,
        'amp_skipped_steps': amp_skipped_steps,
        'amp_scale': scaler.get_scale() if training else None,
    }


def save_checkpoint(path, model, metadata, epoch, stage, metrics):
    payload = {
        **metadata, 'epoch': epoch, 'stage': stage, 'source_val_metrics': metrics,
        'backbone_state_dict': {k: v.detach().cpu() for k, v in model.visual.state_dict().items()},
        'classifier_state_dict': {k: v.detach().cpu() for k, v in model.classifier.state_dict().items()},
    }
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--checkpoint', type=Path, default=BASE / 'checkpoints/RemoteCLIP-RN50.pt')
    parser.add_argument('--output-dir', type=Path, help='New directory; existing directories are refused.')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lp-epochs', type=int, default=5)
    parser.add_argument('--ft-epochs', type=int, default=15)
    parser.add_argument('--lp-lr', type=float, default=1e-3)
    parser.add_argument('--backbone-lr', type=float, default=1e-5)
    parser.add_argument('--head-lr', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--val-ratio', type=float, default=.2)
    parser.add_argument('--seed', type=int, default=924)
    parser.add_argument('--selection-metric', choices=['accuracy', 'macro_f1'], default='accuracy')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, cuda:0, ...')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--check-data', action='store_true', help='Print source split audit without loading model.')
    parser.add_argument('--smoke-test', action='store_true', help='Only one LP and one FT batch, plus validation/save/reload.')
    args = parser.parse_args()
    if not 0 < args.val_ratio < 1:
        parser.error('--val-ratio must be between 0 and 1')
    if args.batch_size < 2 or args.num_workers < 0:
        parser.error('batch size must be >=2 and num workers >=0')
    if min(args.lp_epochs, args.ft_epochs) < 0 or args.lp_epochs + args.ft_epochs == 0:
        parser.error('Use nonnegative epoch counts with at least one training epoch.')
    if min(args.lp_lr, args.backbone_lr, args.head_lr) <= 0 or args.weight_decay < 0:
        parser.error('Learning rates must be positive and weight decay nonnegative.')
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    samples, excluded_dirs, ignored_dirs = collect_samples(args.data_root)
    train_samples, val_samples = split_samples(samples, args.val_ratio, args.seed)
    audit = {
        'excluded_disaster_families': list(EXCLUDED),
        'excluded_directories': excluded_dirs, 'ignored_empty_directories': ignored_dirs,
        'source_disaster_counts': dict(sorted(Counter(s.disaster for s in samples).items())),
        'train_count': len(train_samples), 'val_count': len(val_samples),
        'train_labels': dict(Counter(s.label for s in train_samples)),
        'val_labels': dict(Counter(s.label for s in val_samples)),
    }
    print(json.dumps(audit, indent=2), flush=True)
    if args.check_data:
        return
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    if device.type == 'cpu' and not args.smoke_test:
        print('WARNING: CUDA is unavailable or CPU selected; full training will be slow.', flush=True)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    amp = device.type == 'cuda' and not args.no_amp
    model, preprocess = build_model(args.checkpoint)
    model = model.to(device)
    output_dir = args.output_dir or BASE / 'outputs' / (
        ('smoke_' if args.smoke_test else 'source_') + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        'model_name': 'RN50', 'pretraining': 'RemoteCLIP', 'feature_dim': 1024,
        'feature_location': 'visual_attention_pool_output', 'feature_l2_normalize': False,
        'classifier_type': 'Linear(1024, 2, bias=True)',
        'class_to_label': CLASS_TO_LABEL, 'preprocess': preprocess,
        'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'dataset_audit': audit, 'smoke_test': args.smoke_test,
        'versions': {'torch': str(torch.__version__), 'open_clip': str(open_clip.__version__)},
        'split_method': 'sample-level binary-label stratification; matches trainsource.py',
    }
    (output_dir / 'config.json').write_text(json.dumps(metadata, indent=2) + '\n')
    for split, subset in [('train', train_samples), ('val', val_samples)]:
        with (output_dir / f'{split}_samples.jsonl').open('w') as handle:
            for sample in subset:
                handle.write(json.dumps(asdict(sample)) + '\n')
    transform = make_transform(preprocess)
    # Independent Dataset instances avoid sharing mutable transform settings.
    loader_options = dict(
        batch_size=2 if args.smoke_test else args.batch_size,
        num_workers=0 if args.smoke_test else args.num_workers,
        pin_memory=device.type == 'cuda',
    )
    train_loader = DataLoader(
        XBDDataset(train_samples, transform), shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), **loader_options,
    )
    val_loader = DataLoader(XBDDataset(val_samples, transform), shuffle=False, **loader_options)
    best_score, epoch = -float('inf'), 0
    limit = 1 if args.smoke_test else None
    print(f'Device: {device}; output: {output_dir}; AMP: {amp}', flush=True)
    for stage, epochs in [('lp', args.lp_epochs), ('ft', args.ft_epochs)]:
        if args.smoke_test:
            epochs = 1
        if not epochs:
            continue
        model.visual.requires_grad_(stage == 'ft')
        groups = [{'params': model.classifier.parameters(), 'lr': args.lp_lr if stage == 'lp' else args.head_lr}]
        if stage == 'ft':
            groups.append({'params': model.visual.parameters(), 'lr': args.backbone_lr})
        optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
        scaler = make_stage_scaler(stage, amp)
        print(f'Starting {stage}: fresh GradScaler, scale={scaler.get_scale():g}', flush=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if stage == 'ft' else None
        for _ in range(epochs):
            epoch += 1
            train_metrics = run_epoch(model, train_loader, device, amp, stage, optimizer, scaler, limit)
            val_metrics = run_epoch(model, val_loader, device, amp, stage, max_batches=limit)
            record = {'epoch': epoch, 'stage': stage, 'lr': [g['lr'] for g in optimizer.param_groups],
                      'train': train_metrics, 'val': val_metrics}
            print(json.dumps(record), flush=True)
            with (output_dir / 'history.jsonl').open('a') as handle:
                handle.write(json.dumps(record) + '\n')
            if val_metrics[args.selection_metric] > best_score:
                best_score = val_metrics[args.selection_metric]
                save_checkpoint(output_dir / 'best_source.pt', model, metadata, epoch, stage, val_metrics)
                print(f'Saved best source validation {args.selection_metric}: {best_score:.6f}', flush=True)
            if scheduler:
                scheduler.step()
    save_checkpoint(output_dir / 'final_source.pt', model, metadata, epoch, stage, val_metrics)
    if args.smoke_test:
        saved = torch.load(output_dir / 'final_source.pt', map_location=device, weights_only=True)
        x, _ = next(iter(val_loader))
        model.eval()
        with torch.no_grad():
            expected = model(x.to(device)).clone()
            model.classifier.weight.add_(1.)
            model.visual.load_state_dict(saved['backbone_state_dict'], strict=True)
            model.classifier.load_state_dict(saved['classifier_state_dict'], strict=True)
            torch.testing.assert_close(model(x.to(device)), expected)
        print('SMOKE TEST PASSED: LP/FT, validation, checkpoint round-trip. Not an experiment result.')
    print(f'Finished. Best checkpoint: {output_dir / "best_source.pt"}', flush=True)


if __name__ == '__main__':
    main()
