import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

ARTIFACTS_DIR = 'solution/artifacts'
K = 32

def build_cnn(k=K):
    return nn.Sequential(
        nn.Conv2d(3, k, 3, padding=1), nn.BatchNorm2d(k), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(k, 2*k, 3, padding=1), nn.BatchNorm2d(2*k), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(2*k, 4*k, 3, padding=1), nn.BatchNorm2d(4*k), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(4*k, 4*k, 3, padding=1), nn.BatchNorm2d(4*k), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Dropout(0.3), nn.Linear(4*k, 2),
    )

ckpt = torch.load(f'{ARTIFACTS_DIR}/best_model.pt', map_location='cpu', weights_only=False)
model = build_cnn()
model.load_state_dict(ckpt['state_dict'])
model.eval()
thr = ckpt['threshold']

val = np.load(f'{ARTIFACTS_DIR}/validation.npz')
X, y = val['X'], val['y']

ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long())
loader = DataLoader(ds, batch_size=64, shuffle=False)
all_p = []
with torch.no_grad():
    for xb, yb in loader:
        all_p.append(torch.softmax(model(xb), dim=1)[:, 1])
probs = torch.cat(all_p).numpy()
preds = (probs >= thr).astype(int)

tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0,1]).ravel()
print('=== MODEL PERFORMANCE ===')
print(f'Threshold: {thr:.4f}')
print(f'TP={tp}, TN={tn}, FP={fp}, FN={fn}')
print(f'Recall_AI: {tp/(tp+fn):.4f}, FPR: {fp/(fp+tn):.4f}')
print()

print('=== CONFIDENCE ANALYSIS ===')
for name, mask in [('TP', (y==1)&(preds==1)), ('TN', (y==0)&(preds==0)),
                    ('FP', (y==0)&(preds==1)), ('FN', (y==1)&(preds==0))]:
    p = probs[mask]
    if len(p) > 0:
        print(f'{name} (n={len(p)}): mean={p.mean():.4f}, std={p.std():.4f}, min={p.min():.4f}, max={p.max():.4f}, median={np.median(p):.4f}')
print()

real_probs = probs[y==0]
ai_probs = probs[y==1]
print('=== CLASS SEPARATION ===')
print(f'Real images P(AI): mean={real_probs.mean():.4f}, std={real_probs.std():.4f}')
print(f'AI images P(AI):   mean={ai_probs.mean():.4f}, std={ai_probs.std():.4f}')
print(f'Overlap zone (0.4-0.8): real={((real_probs>=0.4)&(real_probs<=0.8)).sum()}, ai={((ai_probs>=0.4)&(ai_probs<=0.8)).sum()}')
print()

# GradCAM setup
activations_store = [None]
gradients_store = [None]
def save_act(m, i, o): activations_store[0] = o.detach()
def save_grad(m, gi, go): gradients_store[0] = go[0].detach()
model[14].register_forward_hook(save_act)
model[14].register_full_backward_hook(save_grad)

def get_cam(x_tensor):
    x_tensor = x_tensor.requires_grad_(True)
    logits = model(x_tensor)
    model.zero_grad()
    logits[0, 1].backward()
    w = gradients_store[0].mean(dim=(2,3), keepdim=True)
    cam = F.relu((w * activations_store[0]).sum(dim=1)).squeeze().detach().numpy()
    if cam.max() > 0: cam = cam / cam.max()
    return cv2.resize(cam, (64, 64))

np.random.seed(42)
n_sample = 100
real_idx = np.random.choice(np.where(y==0)[0], min(n_sample, int((y==0).sum())), replace=False)
ai_idx = np.random.choice(np.where(y==1)[0], min(n_sample, int((y==1).sum())), replace=False)

real_cams = np.stack([get_cam(torch.from_numpy(X[i:i+1])) for i in real_idx])
ai_cams = np.stack([get_cam(torch.from_numpy(X[i:i+1])) for i in ai_idx])

print('=== GRADCAM SPATIAL ANALYSIS ===')
avg_real = real_cams.mean(axis=0)
avg_ai = ai_cams.mean(axis=0)

center_r = avg_real[16:48, 16:48].mean()
edge_r = (avg_real.sum() - avg_real[16:48, 16:48].sum()) / (64*64 - 32*32)
print(f'Real avg heatmap: center={center_r:.4f}, periphery={edge_r:.4f}, center/periph ratio={center_r/max(edge_r,1e-6):.2f}')

center_a = avg_ai[16:48, 16:48].mean()
edge_a = (avg_ai.sum() - avg_ai[16:48, 16:48].sum()) / (64*64 - 32*32)
print(f'AI avg heatmap:   center={center_a:.4f}, periphery={edge_a:.4f}, center/periph ratio={center_a/max(edge_a,1e-6):.2f}')

print(f'Real avg intensity: {real_cams.mean():.4f} +/- {real_cams.std():.4f}')
print(f'AI avg intensity:   {ai_cams.mean():.4f} +/- {ai_cams.std():.4f}')

corr = np.corrcoef(avg_real.flatten(), avg_ai.flatten())[0,1]
print(f'Correlation between avg real vs AI heatmaps: {corr:.4f}')
print()

fp_idx = np.where((y==0)&(preds==1))[0]
fn_idx = np.where((y==1)&(preds==0))[0]
if len(fp_idx) > 0:
    fp_cams = np.stack([get_cam(torch.from_numpy(X[i:i+1])) for i in fp_idx[:20]])
    print(f'FP heatmap intensity: {fp_cams.mean():.4f} (vs real correct avg {real_cams.mean():.4f})')
if len(fn_idx) > 0:
    fn_cams = np.stack([get_cam(torch.from_numpy(X[i:i+1])) for i in fn_idx[:20]])
    print(f'FN heatmap intensity: {fn_cams.mean():.4f} (vs AI correct avg {ai_cams.mean():.4f})')

print()
print('=== QUADRANT ATTENTION ===')
for name, hm in [('Real', avg_real), ('AI', avg_ai)]:
    tl = hm[:32,:32].mean(); tr = hm[:32,32:].mean()
    bl = hm[32:,:32].mean(); br = hm[32:,32:].mean()
    print(f'{name}: TL={tl:.4f} TR={tr:.4f} BL={bl:.4f} BR={br:.4f}')
