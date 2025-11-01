import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter
from IPython.display import display
from ptflops import get_model_complexity_info
from thop import profile
import glob

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def compute_model_stats(model: nn.Module, input_shape=(9, 128)):
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_shape).to(device)
    model.eval()
    classifier_original = model.classifier
    model.classifier = nn.Identity()
    macs_backbone, params_backbone = profile(model, inputs=(dummy_input,), verbose=False)
    model.classifier = classifier_original
    with torch.no_grad():
        features = model.extract_features(dummy_input)
        if isinstance(features, tuple):
            features = features[0]
    macs_cls, params_cls = profile(model.classifier, inputs=(features,), verbose=False)
    total_macs_m = (macs_backbone + macs_cls) / 1e6
    total_params_m = (params_backbone + params_cls) / 1e6
    return total_params_m, total_macs_m

def measure_inference_time(model: nn.Module, input_size=(1, 9, 128), device='cuda', n_runs=100, warmup=10):
    model.eval()
    inputs = torch.randn(input_size).to(device)
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(inputs)
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    with torch.inference_mode():
        for _ in range(n_runs):
            _ = model(inputs)
    if device == 'cuda':
        torch.cuda.synchronize()
    end = time.time()
    return (end - start) / n_runs * 1000

def plot_confusion_matrix(cm, labels, save_path='confusion_matrix.png'):
    cm_norm = cm.astype('float') / np.sum(cm, axis=1)[:, np.newaxis]
    labels_wrapped = [l.replace(' ', '\n') for l in labels]
    df = pd.DataFrame(cm_norm, index=labels_wrapped, columns=labels_wrapped)
    annot = df.copy().astype(str)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            annot.iloc[i, j] = f"{df.iloc[i, j]:.2f}"
    plt.figure(figsize=(6, 6))
    sns.heatmap(df, annot=annot.values, fmt="", cmap="Blues", cbar=True, annot_kws={"size": 9}, vmin=0, vmax=1)
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_edgecolor('black')
    plt.xlabel('Predicted Label', fontsize=10)
    plt.ylabel('True Label', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

class HARDataset(Dataset):
    def __init__(self, X, y, normalize=True):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        if normalize:
            self.mean = self.X.mean(axis=(0, 2), keepdims=True)
            self.std = self.X.std(axis=(0, 2), keepdims=True) + 1e-8
            self.X = (self.X - self.mean) / self.std
        else:
            self.mean, self.std = 0, 1
    def set_stats(self, mean, std):
        self.mean, self.std = mean, std
        self.X = (self.X - self.mean) / self.std
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), int(self.y[idx])

def read_txt_matrix(file_path):
    return np.loadtxt(file_path, dtype=np.float32)

def load_full_uci_har_data(root: str):
    UCI_CHANNELS_PREFIX = [
        "total_acc_x_", "total_acc_y_", "total_acc_z_",
        "body_acc_x_", "body_acc_y_", "body_acc_z_",
        "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
    ]
    def load_split(split):
        channels = [read_txt_matrix(os.path.join(root, f"{prefix}{split}.txt")) for prefix in UCI_CHANNELS_PREFIX]
        X = np.stack(channels, axis=1)
        y = read_txt_matrix(os.path.join(root, f"y_{split}.txt")).astype(int) - 1
        subjects = read_txt_matrix(os.path.join(root, f"subject_{split}.txt")).astype(int)
        return X, y, subjects
    X_train, y_train, subjects_train = load_split("train")
    X_test, y_test, subjects_test = load_split("test")
    X_all = np.concatenate((X_train, X_test), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)
    subjects_all = np.concatenate((subjects_train, subjects_test), axis=0)
    sorted_indices = np.argsort(subjects_all)
    X_all, y_all, subjects_all = X_all[sorted_indices], y_all[sorted_indices], subjects_all[sorted_indices]
    activity_names = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Laying']
    return X_all, y_all, subjects_all, activity_names

def fft_filter(x, cutoff_hz, fs, btype='low'):
    B, C, T = x.shape
    freqs = torch.fft.fftfreq(T, d=1/fs).to(x.device)
    x_fft = torch.fft.fft(x, dim=-1)
    if btype == 'low':
        mask = torch.abs(freqs) <= cutoff_hz
    else:
        mask = torch.abs(freqs) > cutoff_hz
    mask = mask.view(1, 1, -1).expand(B, C, -1)
    x_fft_filtered = x_fft * mask
    return torch.fft.ifft(x_fft_filtered, dim=-1).real

def compute_gravity_gyro_consistency(total_acc, gyro, gravity_est, fs, eps=1e-6):
    dt = 1.0 / fs
    B, C, T = total_acc.shape
    num_sensors = min(total_acc.shape[1] // 3, gyro.shape[1] // 3)
    total_sensor_loss = 0.0
    ux = torch.tensor([1., 0., 0.], device=total_acc.device)
    uy = torch.tensor([0., 1., 0.], device=total_acc.device)
    uz = torch.tensor([0., 0., 1.], device=total_acc.device)
    for i in range(num_sensors):
        start_idx, end_idx = i * 3, (i + 1) * 3
        sensor_gravity_est = gravity_est[:, start_idx:end_idx, :]
        sensor_gyro = gyro[:, start_idx:end_idx, :]
        gravity_norm = F.normalize(sensor_gravity_est, dim=1, eps=eps)
        loss_per_sensor = 0.0
        for t in range(1, T):
            gravity_prev = gravity_norm[:, :, t - 1]
            gravity_curr = gravity_norm[:, :, t]
            gyro_angular_vel = sensor_gyro[:, :, t - 1] * dt
            gravity_predicted = gravity_prev.clone()
            axis_x = torch.cross(gravity_prev, ux.expand_as(gravity_prev), dim=1)
            axis_y = torch.cross(gravity_prev, uy.expand_as(gravity_prev), dim=1)
            axis_z = torch.cross(gravity_prev, uz.expand_as(gravity_prev), dim=1)
            rotation_x = gyro_angular_vel[:, 0:1]
            rotation_y = gyro_angular_vel[:, 1:2]
            rotation_z = gyro_angular_vel[:, 2:3]
            gravity_predicted = gravity_predicted + rotation_x * axis_x + rotation_y * axis_y + rotation_z * axis_z
            gravity_predicted = F.normalize(gravity_predicted, dim=1, eps=eps)
            consistency_loss = F.mse_loss(gravity_predicted, gravity_curr)
            loss_per_sensor += consistency_loss
        total_sensor_loss += loss_per_sensor / (T - 1)
    return total_sensor_loss / num_sensors

def physics_guided_loss(x, gravity_scale, fs=50, hp_cut=1.5, lp_cut=0.5, eps=1e-6):
    acc_indices = [3, 4, 5]
    gyro_indices = [6, 7, 8]
    total_acc = x[:, acc_indices, :]
    gyro = x[:, gyro_indices, :]
    low_freq = fft_filter(total_acc, cutoff_hz=lp_cut, fs=fs, btype='low')
    gravity_mag = torch.norm(low_freq, dim=1).mean(dim=-1)
    L_grav = compute_gravity_gyro_consistency(total_acc, gyro, low_freq, fs)
    acc_high = fft_filter(total_acc, cutoff_hz=hp_cut, fs=fs, btype='high')
    gyro_high = fft_filter(gyro, cutoff_hz=hp_cut, fs=fs, btype='high')
    acc_activity = (acc_high ** 2).sum(dim=1)
    gyro_activity = (gyro_high ** 2).sum(dim=1)
    acc_norm = (acc_activity - acc_activity.mean(dim=-1, keepdims=True)) / (acc_activity.std(dim=-1, keepdims=True) + eps)
    gyro_norm = (gyro_activity - gyro_activity.mean(dim=-1, keepdims=True)) / (gyro_activity.std(dim=-1, keepdims=True) + eps)
    L_ag = F.mse_loss(acc_norm, gyro_norm)
    acc_temporal = acc_high.flatten(1)
    gyro_temporal = gyro_high.flatten(1)
    acc_t_norm = F.normalize(acc_temporal, dim=1)
    gyro_t_norm = F.normalize(gyro_temporal, dim=1)
    correlation = (acc_t_norm * gyro_t_norm).sum(dim=1).mean()
    L_ag_corr = 1.0 - correlation
    L_ag_combined = L_ag + 2.0 * L_ag_corr
    acc_jerk = torch.diff(total_acc, dim=-1)
    gyro_jerk = torch.diff(gyro, dim=-1)
    L_jerk = (acc_jerk ** 2).mean() + (gyro_jerk ** 2).mean()
    return L_grav, L_ag_combined, L_jerk

class ODEF(nn.Module):
    def forward_with_grad(self, z, t, grad_outputs):
        batch_size = z.shape[0]
        out = self.forward(z, t)
        a = grad_outputs
        adfdz, adfdt, *adfdp = torch.autograd.grad(
            (out,), (z, t) + tuple(self.parameters()), grad_outputs=(a),
            allow_unused=True, retain_graph=True
        )
        if adfdp is not None:
            adfdp = torch.cat([p_grad.flatten() for p_grad in adfdp]).unsqueeze(0)
            adfdp = adfdp.expand(batch_size, -1) / batch_size
        if adfdt is not None:
            adfdt = adfdt.expand(batch_size, 1) / batch_size
        return out, adfdz, adfdt, adfdp
    def flatten_parameters(self):
        p_shapes = []
        flat_parameters = []
        for p in self.parameters():
            p_shapes.append(p.size())
            flat_parameters.append(p.flatten())
        return torch.cat(flat_parameters)

class ODEAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z0, t, flat_parameters, func):
        assert isinstance(func, ODEF)
        bs, *z_shape = z0.size()
        time_len = t.size(0)
        with torch.no_grad():
            z = torch.zeros(time_len, bs, *z_shape).to(z0)
            z[0] = z0
            for i_t in range(time_len - 1):
                z0 = ode_solve(z0, t[i_t], t[i_t+1], func)
                z[i_t+1] = z0
        ctx.func = func
        ctx.save_for_backward(t, z.clone(), flat_parameters)
        return z
    @staticmethod
    def backward(ctx, dLdz):
        func = ctx.func
        t, z, flat_parameters = ctx.saved_tensors
        time_len, bs, *z_shape = z.size()
        n_dim = np.prod(z_shape)
        n_params = flat_parameters.size(0)
        def augmented_dynamics(aug_z_i, t_i):
            z_i, a = aug_z_i[:, :n_dim], aug_z_i[:, n_dim:2*n_dim]
            z_i = z_i.view(bs, *z_shape).float()
            a = a.view(bs, *z_shape).float()
            t_i = t_i.float()
            with torch.set_grad_enabled(True):
                t_i = t_i.detach().requires_grad_(True)
                z_i = z_i.detach().requires_grad_(True)
                func_eval, adfdz, adfdt, adfdp = func.forward_with_grad(z_i, t_i, grad_outputs=a)
                adfdz = adfdz.to(z_i) if adfdz is not None else torch.zeros(bs, *z_shape).to(z_i)
                adfdp = adfdp.to(z_i) if adfdp is not None else torch.zeros(bs, n_params).to(z_i)
                adfdt = adfdt.to(z_i) if adfdt is not None else torch.zeros(bs, 1).to(z_i)
            func_eval = func_eval.view(bs, n_dim)
            adfdz = adfdz.view(bs, n_dim)
            return torch.cat((func_eval, -adfdz, -adfdp, -adfdt), dim=1)
        dLdz = dLdz.view(time_len, bs, n_dim).float()
        with torch.no_grad():
            adj_z = torch.zeros(bs, n_dim, dtype=torch.float32, device=dLdz.device)
            adj_p = torch.zeros(bs, n_params, dtype=torch.float32, device=dLdz.device)
            adj_t = torch.zeros(time_len, bs, 1, dtype=torch.float32, device=dLdz.device)
            for i_t in range(time_len-1, 0, -1):
                z_i = z[i_t].float()
                t_i = t[i_t].float()
                f_i = func(z_i, t_i).view(bs, n_dim).float()
                dLdz_i = dLdz[i_t].float()
                dLdt_i = torch.bmm(torch.transpose(dLdz_i.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]
                adj_z += dLdz_i
                adj_t[i_t] = adj_t[i_t] - dLdt_i
                aug_z = torch.cat((z_i.view(bs, n_dim), adj_z, torch.zeros(bs, n_params, dtype=torch.float32, device=z.device), adj_t[i_t]), dim=-1)
                aug_ans = ode_solve(aug_z, t_i, t[i_t-1], augmented_dynamics)
                adj_z[:] = aug_ans[:, n_dim:2*n_dim]
                adj_p[:] += aug_ans[:, 2*n_dim:2*n_dim + n_params]
                adj_t[i_t-1] = aug_ans[:, 2*n_dim + n_params:]
                del aug_z, aug_ans
            dLdz_0 = dLdz[0].float()
            dLdt_0 = torch.bmm(torch.transpose(dLdz_0.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]
            adj_z += dLdz_0
            adj_t[0] = adj_t[0] - dLdt_0
        return adj_z.view(bs, *z_shape), adj_t, adj_p, None

class NeuralODE(nn.Module):
    def __init__(self, func):
        super(NeuralODE, self).__init__()
        assert isinstance(func, ODEF)
        self.func = func
    def forward(self, z0, t=torch.Tensor([0., 1.]), return_whole_sequence=False):
        t = t.to(z0)
        z = ODEAdjoint.apply(z0, t, self.func.flatten_parameters(), self.func)
        if return_whole_sequence:
            return z
        else:
            return z[-1]

def ode_solve(z0, t0, t1, f):
    h_max = 0.05
    n_steps = math.ceil((abs(t1 - t0)/h_max).max().item())
    h = (t1 - t0)/n_steps
    t = t0
    z = z0
    for i_step in range(n_steps):
        z = z + h * f(z, t)
        t = t + h
    return z

class NNODEF(ODEF):
    def __init__(self, in_dim, hid_dim, time_invariant=False):
        super(NNODEF, self).__init__()
        self.time_invariant = time_invariant
        if time_invariant:
            self.lin1 = nn.Linear(in_dim, hid_dim)
        else:
            self.lin1 = nn.Linear(in_dim+1, hid_dim)
        self.lin2 = nn.Linear(hid_dim, hid_dim)
        self.lin3 = nn.Linear(hid_dim, in_dim)
        self.elu = nn.ELU(inplace=True)
    def forward(self, x, t):
        if not self.time_invariant:
            t = t.expand(x.shape[0], 1)
            x = torch.cat((x, t), dim=-1)
        h = self.elu(self.lin1(x))
        h = self.elu(self.lin2(h))
        out = self.lin3(h)
        return out

class UCIHarODEModel(nn.Module):
    def __init__(self, n_channels=9, seq_length=128, d_model=64, n_classes=6, **kwargs):
        super().__init__()
        self.proj = nn.Conv1d(n_channels, d_model, 1)
        self.func = NNODEF(d_model, d_model * 2, time_invariant=False)
        self.ode = NeuralODE(self.func)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)
        self.gravity_scale = nn.Parameter(torch.tensor(1.0))
        
    def extract_features(self, x):
        x = self.proj(x)
        B, D, T = x.shape
        h = x.transpose(1, 2)
        
        z0 = h.mean(dim=1)
        
        t = torch.tensor([0., 1.], device=x.device)
        
        z1 = self.ode(z0, t, return_whole_sequence=False) 
        
        features = self.norm(z1.float())
        return features
        
    def forward(self, x):
        return self.classifier(self.extract_features(x))

def evaluate_model(model, loader, device='cuda'):
    model.eval()
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            output = model(x)
            preds = output.argmax(dim=-1) if isinstance(output, torch.Tensor) else output[0].argmax(dim=-1)
            all_preds.append(preds.cpu().numpy()); all_labels.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds); all_labels = np.concatenate(all_labels)
    return {'accuracy': accuracy_score(all_labels, all_preds), 'f1_macro': f1_score(all_labels, all_preds, average='macro'),
            'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
            'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(all_labels, all_preds)}

def print_results(exp_name, metrics, params, flops, inference_time, activity_names):
    print(f"\n{'='*80}\nResults: {exp_name}\n{'='*80}")
    print(f"Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f} | Prec: {metrics['precision_macro']:.4f} | Rec: {metrics['recall_macro']:.4f}")
    print(f"Params: {params:.4f}M | FLOPs: {flops:.4f}M | Inference: {inference_time:.4f}ms\n{'='*80}\n")
    cm_path = f"./cm_{exp_name.replace(' ', '_')}.png"
    plot_confusion_matrix(metrics['confusion_matrix'], activity_names, cm_path)

def train_model(model, train_loader, val_loader, epochs, lr, weight_decay, device, use_physics_loss, use_group_ortho, train_mean, train_std, model_args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler('cuda')
    scheduler = None
    if epochs > model_args['warmup_epochs']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - model_args['warmup_epochs'])
    mean_t = torch.from_numpy(train_mean).to(device)
    std_t = torch.from_numpy(train_std).to(device)
    best_val_acc, best_state = 0.0, None
    for epoch in range(1, epochs + 1):

        print (epoch)
        if epoch <= model_args['warmup_epochs']:
            for pg in optimizer.param_groups:
                pg["lr"] = lr * epoch / model_args['warmup_epochs']
        elif scheduler:
            scheduler.step()
        if epoch <= model_args['physics_warmup_epochs']:
            progress = epoch / model_args['physics_warmup_epochs']
            physics_factor = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        else:
            physics_factor = 1.0
        model.train()
        total_loss, loss_ce_sum = 0.0, 0.0
        correct, total = 0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                output = model(x)
                logits = output
                loss_ce = F.cross_entropy(logits, y, label_smoothing=model_args['label_smoothing'])
                loss = loss_ce
                loss_ce_sum += loss_ce.item() * y.size(0)
                if use_physics_loss and hasattr(model, 'gravity_scale'):
                    x_raw = x * std_t + mean_t
                    L_grav, L_ag, L_jerk = physics_guided_loss(x_raw, model.gravity_scale)
                    loss_phys = physics_factor * (model_args['w_grav'] * L_grav + model_args['w_ag'] * L_ag + model_args['w_jerk'] * L_jerk)
                    loss += loss_phys
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.size(0)
        val_metrics = evaluate_model(model, val_loader, device)
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        if int(epoch / epochs) == 2:
            print("1/2")
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model

def run_experiment(exp_name, model_class, model_args, train_loader, val_loader, test_loader, epochs, device, activity_names, train_mean, train_std, results_list, use_physics_loss, use_group_ortho):
    print(f"\n{'#'*80}\n# Running: {exp_name}\n{'#'*80}")
    model = model_class(**model_args).to(device)
    model = train_model(model, train_loader, val_loader, epochs, model_args['lr'], model_args['weight_decay'], device, use_physics_loss, use_group_ortho, train_mean, train_std, model_args)
    metrics = evaluate_model(model, test_loader, device)
    params, flops = compute_model_stats(model)
    inference_time = measure_inference_time(model, device=device)
    print_results(exp_name, metrics, params, flops, inference_time, activity_names)
    results_list.append({'Experiment': exp_name, 'Accuracy': f"{metrics['accuracy']:.4f}", 'F1': f"{metrics['f1_macro']:.4f}",
                         'Precision': f"{metrics['precision_macro']:.4f}", 'Recall': f"{metrics['recall_macro']:.4f}",
                         'Params(M)': f"{params:.4f}", 'FLOPs(M)': f"{flops:.4f}", 'Inference(ms)': f"{inference_time:.4f}"})
    return model

def add_realistic_noise(X_original, noise_type, noise_ratio, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X_noisy = X_original.copy()
    if noise_ratio <= 0:
        return X_noisy
    if noise_type == 'jitter':
        B, C, T = X_original.shape
        for b in range(B):
            for c in range(C):
                jitter_std = noise_ratio * 50
                time_noise = np.random.normal(0, jitter_std, T)
                time_indices = np.arange(T) + time_noise
                time_indices = np.clip(time_indices, 0, T-1)
                interp_values = np.interp(time_indices, np.arange(T), X_original[b, c, :])
                X_noisy[b, c, :] = interp_values.astype(np.float32)
    elif noise_type == 'dropout':
        dropout_mask = np.random.random(X_original.shape) < noise_ratio
        X_noisy[dropout_mask] = 0.0
    elif noise_type == 'window_shifting':
        B, C, T = X_original.shape
        window_size = max(1, int(T * 0.3))
        max_shift = max(1, int(noise_ratio * T * 0.8))
        for b in range(B):
            for c in range(C):
                num_windows = T // window_size
                for w in range(num_windows):
                    start_idx = w * window_size
                    end_idx = min((w + 1) * window_size, T)
                    if np.random.random() < noise_ratio:
                        shift = np.random.randint(-max_shift, max_shift + 1)
                        if shift > 0:
                            X_noisy[b, c, start_idx:end_idx-shift] = X_original[b, c, start_idx+shift:end_idx]
                            X_noisy[b, c, end_idx-shift:end_idx] = X_original[b, c, end_idx-1]
                        elif shift < 0:
                            X_noisy[b, c, start_idx-shift:end_idx] = X_original[b, c, start_idx:end_idx+shift]
                            X_noisy[b, c, start_idx:start_idx-shift] = X_original[b, c, start_idx]
    return X_noisy

if __name__ == "__main__":
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    dataset_root = "/content/drive/MyDrive/HAR_Dataset/UCI"
    epochs = 100
    batch_size = 512
    num_workers = 1
    base_model_args = {
        'n_channels': 9,
        'seq_length': 128,
        'd_model': 64,
        'n_classes': 6,
        'lr': 1e-3,
        'weight_decay': 1e-2,
        'label_smoothing': 0.05,
        'warmup_epochs': 5,
        'physics_warmup_epochs': 10,
        'w_grav': 0.05,
        'w_ag': 0.05,
        'w_jerk': 0.01,
        'w_ortho': 0.01
    }
    X_all, y_all, subjects_all, activity_names = load_full_uci_har_data(dataset_root)
    if X_all is None:
        print("Failed to load UCI HAR data. Please check the dataset directory and file contents.")
        exit()
    test_subjects = [3, 12, 13, 15, 16, 17, 21, 26, 27]
    train_mask = ~np.isin(subjects_all, test_subjects)
    test_mask = np.isin(subjects_all, test_subjects)
    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    X_test = X_all[test_mask]
    y_test = y_all[test_mask]
    results_list = []
    model_configs = [
        {'name': 'ODE_Physics', 'class': UCIHarODEModel, 'args': {**base_model_args}, 'use_physics': True, 'use_ortho': False},
    ]
    trained_models = {}
    normalization_stats = None
    for config in model_configs:
        try:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            train_idx, val_idx = next(kf.split(X_train, y_train))
            train_ds = HARDataset(X_train[train_idx], y_train[train_idx], normalize=True)
            val_ds = HARDataset(X_train[val_idx], y_train[val_idx], normalize=False)
            test_ds = HARDataset(X_test, y_test, normalize=False)
            val_ds.set_stats(train_ds.mean, train_ds.std)
            test_ds.set_stats(train_ds.mean, train_ds.std)
            normalization_stats = (train_ds.mean, train_ds.std)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                    num_workers=num_workers, pin_memory=True,
                                    prefetch_factor=2, persistent_workers=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                   num_workers=num_workers, pin_memory=True,
                                   prefetch_factor=2, persistent_workers=True)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=True,
                                    prefetch_factor=2, persistent_workers=True)
            current_train_mean, current_train_std = train_ds.mean, train_ds.std
            model = run_experiment(config['name'], config['class'], config['args'], train_loader, val_loader, test_loader, epochs,
                                  device, activity_names, current_train_mean, current_train_std, results_list,
                                  config['use_physics'], config['use_ortho'])
            trained_models[config['name']] = model
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue
    if len(trained_models) == 0:
        exit()
    noise_types = ['jitter', 'dropout', 'window_shifting']
    display_labels = ['0%', '5%', '10%', '15%', '20%', '25%', '30%']
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    for noise_type in noise_types:
        noise_results = []
        for level, label in zip(noise_levels, display_labels):
            row = {'Noise_Level': label}
            for model_name, model in trained_models.items():
                X_test_noisy = add_realistic_noise(X_test, noise_type, level, seed=42)
                test_ds_noisy = HARDataset(X_test_noisy, y_test, normalize=False)
                test_ds_noisy.set_stats(normalization_stats[0], normalization_stats[1])
                test_loader_noisy = DataLoader(test_ds_noisy, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True,
                                              prefetch_factor=2, persistent_workers=True)
                metrics = evaluate_model(model, test_loader_noisy, device)
                row[f'{model_name}_Acc'] = f"{metrics['accuracy']:.4f}"
                row[f'{model_name}_F1'] = f"{metrics['f1_macro']:.4f}"
            noise_results.append(row)
        retention_row = {'Noise_Level': 'Avg_Retention'}
        for model_name in trained_models.keys():
            accuracies = [float(row[f'{model_name}_Acc']) for row in noise_results[1:]]
            base_acc = float(noise_results[0][f'{model_name}_Acc'])
            if base_acc > 0:
                retentions = [acc / base_acc for acc in accuracies]
                mean_retention = np.mean(retentions)
                retention_row[f'{model_name}_Acc'] = f"{mean_retention:.4f}"
                retention_row[f'{model_name}_F1'] = '-'
            else:
                retention_row[f'{model_name}_Acc'] = 'ERROR'
                retention_row[f'{model_name}_F1'] = 'ERROR'
        noise_results.append(retention_row)
        noise_df = pd.DataFrame(noise_results)
        display(noise_df)
        noise_df.to_csv(f'./{noise_type}_noise_robustness_results.csv', index=False)
    rst_df = pd.DataFrame(results_list)
    display(rst_df)
    rst_df.to_csv('./model_comparison_results.csv', index=False)
