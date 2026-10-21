import math
import numpy as np
METRIC_KEYS = ['local_test_acc', 'global_balanced_test_acc', 'client_unseen_class_acc', 'rare_class_acc', 'l25', 'u25', 'r25', 'f25']
F25_FRACTION = 0.25

def safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result

def scalar_accuracy(correct, total):
    total = float(total)
    if total <= 0:
        return float('nan')
    return float(correct) / total

def balanced_accuracy(per_class_correct, per_class_total, class_mask=None):
    per_class_correct = np.asarray(per_class_correct, dtype=np.float64)
    per_class_total = np.asarray(per_class_total, dtype=np.float64)
    recalls = safe_divide(per_class_correct, per_class_total)
    valid_mask = per_class_total > 0
    if class_mask is not None:
        valid_mask = np.logical_and(valid_mask, np.asarray(class_mask, dtype=bool))
    if not np.any(valid_mask):
        return float('nan')
    return float(np.nanmean(recalls[valid_mask]))

def cvar_worst_fraction_mean(values, fraction=F25_FRACTION, count_base=None):
    values = np.asarray(values, dtype=np.float64).ravel()
    base = len(values) if count_base is None else int(count_base)
    defined = values[~np.isnan(values)]
    if len(defined) == 0:
        return float('nan')
    count = max(1, int(math.ceil(base * fraction)))
    count = min(count, len(defined))
    return float(np.mean(np.sort(defined)[:count]))

def per_client_floor(local_acc, unseen_acc=None, rare_acc=None):
    local_acc = np.asarray(local_acc, dtype=np.float64).ravel()
    num_clients = local_acc.shape[0]

    def _as_axis(values, name):
        if values is None:
            return np.full(num_clients, np.nan, dtype=np.float64)
        axis = np.asarray(values, dtype=np.float64).ravel()
        if axis.shape[0] != num_clients:
            raise ValueError(f'{name} must have one entry per client ({num_clients})')
        return axis
    stacked = np.vstack([local_acc, _as_axis(unseen_acc, 'unseen_acc'), _as_axis(rare_acc, 'rare_acc')])
    defined = ~np.isnan(stacked)
    floor = np.where(defined, stacked, np.inf).min(axis=0)
    all_undefined = ~defined.any(axis=0)
    floor[all_undefined] = local_acc[all_undefined]
    return floor

def personalization_floor_panel(local_acc, unseen_acc=None, rare_acc=None, fraction=F25_FRACTION):
    local_acc = np.asarray(local_acc, dtype=np.float64).ravel()
    num_clients = local_acc.shape[0]
    floor = per_client_floor(local_acc, unseen_acc=unseen_acc, rare_acc=rare_acc)
    unseen_axis = np.full(num_clients, np.nan, dtype=np.float64) if unseen_acc is None else np.asarray(unseen_acc, dtype=np.float64).ravel()
    rare_axis = np.full(num_clients, np.nan, dtype=np.float64) if rare_acc is None else np.asarray(rare_acc, dtype=np.float64).ravel()
    return {'f25': cvar_worst_fraction_mean(floor, fraction=fraction, count_base=num_clients), 'l25': cvar_worst_fraction_mean(local_acc, fraction=fraction, count_base=num_clients), 'u25': cvar_worst_fraction_mean(unseen_axis, fraction=fraction, count_base=num_clients), 'r25': cvar_worst_fraction_mean(rare_axis, fraction=fraction, count_base=num_clients), 'u25_defined_clients': int(np.count_nonzero(~np.isnan(unseen_axis))), 'num_clients': int(num_clients), 'tail_size': max(1, int(math.ceil(num_clients * fraction))), 'per_client_floor': floor}

def per_client_rare_mask(train_label_counts, rare_k=2):
    counts = np.asarray(train_label_counts)
    if counts.ndim != 2:
        raise ValueError('train_label_counts must have shape (num_clients, num_classes)')
    if rare_k < 1:
        raise ValueError('rare_k must be at least 1')
    mask = np.zeros(counts.shape, dtype=bool)
    for (client_idx, client_counts) in enumerate(counts):
        present = np.flatnonzero(client_counts > 0)
        order = np.lexsort((present, client_counts[present]))
        mask[client_idx, present[order[:rare_k]]] = True
    return mask
