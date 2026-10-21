import torch
import os
import numpy as np
import h5py
import copy
import time
import random
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data, read_global_test_data
from utils.metrics import METRIC_KEYS, balanced_accuracy, per_client_rare_mask, personalization_floor_panel, safe_divide, scalar_accuracy
from utils.output_utils import (
    DETAIL_KIND, SUMMARY_KIND, RESULT_FORMAT_VERSION,
    flat_model_basename,
    result_file_path, current_timestamp, build_result_metadata,
)

class Server(object):

    def __init__(self, args, times):
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []
        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []
        self.rs_local_test_acc = []
        self.rs_global_balanced_test_acc = []
        self.rs_client_unseen_class_acc = []
        self.rs_rare_class_acc = []
        self.rs_total_comm_bytes = []
        self.rs_f25 = []
        self.rs_l25 = []
        self.rs_u25 = []
        self.rs_r25 = []
        self.comm_bytes_total = 0
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate
        self.saved_result_path = None
        self.saved_model_path = None
        self.save_eval_detail = getattr(args, 'save_eval_detail', True)
        self.saved_eval_detail_path = None
        self.result_timestamp = None
        self.eval_detail_static_profiles = None
        self.current_eval_round_idx = None
        self.current_eval_stage_code = None
        self.eval_stage_codes = {'pre_local_train': 0, 'post_train_final': 1}
        self.skip_next_detail_record = False
        self.global_test_data = None
        self.global_test_loader = None
        self.global_test_label_counts = None
        self.global_test_data = read_global_test_data(self.dataset)
        if self.global_test_data is not None and len(self.global_test_data) > 0:
            self.global_test_loader = DataLoader(self.global_test_data, batch_size=self.batch_size, drop_last=False, shuffle=False)
            self.global_test_label_counts = self.count_labels(self.global_test_data)

    def set_clients(self, clientObj):
        for (i, train_slow, send_slow) in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, id=i, train_samples=len(train_data), test_samples=len(test_data), train_slow=train_slow, send_slow=send_slow)
            self.clients.append(client)

    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True
        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients + 1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))
        return selected_clients

    def send_models(self):
        assert len(self.clients) > 0
        for client in self.clients:
            start_time = time.time()
            client.set_parameters(self.global_model)
            self.add_comm_bytes(self.payload_num_bytes(self.global_model))
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert len(self.selected_clients) > 0
        active_clients = random.sample(self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
                self.add_comm_bytes(self.payload_num_bytes(client.model))
        for (i, w) in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def add_comm_bytes(self, byte_count):
        self.comm_bytes_total += int(byte_count)

    def payload_num_bytes(self, payload):
        if payload is None:
            return 0
        if isinstance(payload, torch.nn.Module):
            return sum((self.payload_num_bytes(tensor) for tensor in payload.state_dict().values()))
        if torch.is_tensor(payload):
            return payload.numel() * payload.element_size()
        if isinstance(payload, dict):
            return sum((self.payload_num_bytes(item) for item in payload.values()))
        if isinstance(payload, (list, tuple)):
            return sum((self.payload_num_bytes(item) for item in payload))
        if isinstance(payload, (int, float, np.integer, np.floating)):
            return np.asarray(payload).nbytes
        return 0

    def aggregate_parameters(self):
        assert len(self.uploaded_models) > 0
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
        for buf in self.global_model.buffers():
            if buf.is_floating_point():
                buf.data.zero_()
        for (w, client_model) in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for (server_param, client_param) in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w
        for (server_buf, client_buf) in zip(self.global_model.buffers(), client_model.buffers()):
            if server_buf.is_floating_point() and server_buf.shape == client_buf.shape:
                server_buf.data += client_buf.data.clone() * w

    def _flat_model_path(self, kind):
        model_path = self.args.run_dir
        base = flat_model_basename(self.args)
        return os.path.join(model_path, f'{base}_{kind}.pt')

    def save_global_model(self):
        self.saved_model_path = self._flat_model_path('terminal')
        torch.save(self.global_model, self.saved_model_path)


    def save_results(self):
        if len(self.rs_test_acc):
            file_path = self.get_summary_result_file_path()
            self.saved_result_path = file_path
            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                for metric_key in METRIC_KEYS:
                    curve = getattr(self, f'rs_{metric_key}', None)
                    if curve is not None:
                        hf.create_dataset(f'rs_{metric_key}', data=curve)
                self.write_metadata_group(hf, SUMMARY_KIND, os.path.basename(file_path))

    def set_eval_context(self, round_idx, stage_name='pre_local_train'):
        self.current_eval_round_idx = int(round_idx)
        self.current_eval_stage_code = int(self.eval_stage_codes[stage_name])

    def clear_eval_context(self):
        self.current_eval_round_idx = None
        self.current_eval_stage_code = None

    def count_labels(self, data, num_classes=None):
        total_classes = self.args.num_classes if num_classes is None else num_classes
        label_counts = [0 for _ in range(total_classes)]
        for (_, y) in data:
            label_counts[int(y.item())] += 1
        return label_counts

    def infer_eval_round_idx(self, series_length):
        return int(series_length * self.eval_gap)

    def infer_eval_stage_code(self):
        return int(self.eval_stage_codes['pre_local_train'])

    def ensure_eval_detail_static_profiles(self):
        if self.eval_detail_static_profiles is not None:
            return
        self.eval_detail_static_profiles = []
        for client in self.clients:
            train_data = read_client_data(self.dataset, client.id, is_train=True)
            test_data = read_client_data(self.dataset, client.id, is_train=False)
            self.eval_detail_static_profiles.append({'client_id': client.id, 'train_samples': client.train_samples, 'test_samples': client.test_samples, 'train_label_counts': self.count_labels(train_data), 'test_label_counts': self.count_labels(test_data)})

    def get_eval_detail_file_path(self):
        return result_file_path(self.args, DETAIL_KIND)

    def ensure_result_timestamp(self):
        if self.result_timestamp is None:
            self.result_timestamp = current_timestamp()

    def get_summary_result_file_path(self):
        return result_file_path(self.args, SUMMARY_KIND)

    def _ensure_root_groups(self, hf):
        attrs_group = hf.require_group('attrs')
        static_group = hf.require_group('static')
        eval_group = hf.require_group('eval')
        return (attrs_group, static_group, eval_group)

    def _ensure_eval_detail_attr_group(self, attrs_group, records_generalization):
        attrs_group.attrs['format_version'] = RESULT_FORMAT_VERSION
        attrs_group.attrs['algorithm'] = self.algorithm
        attrs_group.attrs['dataset'] = self.dataset
        attrs_group.attrs['goal'] = self.goal
        attrs_group.attrs['seed'] = int(self.args.seed)
        attrs_group.attrs['global_rounds'] = int(self.global_rounds)
        attrs_group.attrs['local_epochs'] = int(self.local_epochs)
        attrs_group.attrs['timestamp'] = self.result_timestamp
        attrs_group.attrs['eval_semantics'] = 'per_event_stage'
        attrs_group.attrs['eval_stage_mapping'] = ','.join(f'{code}={name}' for name, code in self.eval_stage_codes.items())
        attrs_group.attrs['has_global_test'] = bool(self.global_test_loader is not None)
        attrs_group.attrs['records_generalization'] = bool(records_generalization)

    def write_metadata_group(self, hf, artifact_kind, file_name):
        self.ensure_result_timestamp()
        metadata = build_result_metadata(self.args, artifact_kind=artifact_kind, file_name=file_name, timestamp=self.result_timestamp)
        metadata_group = hf.require_group('metadata')
        for (key, value) in metadata.items():
            if key == 'args_json':
                continue
            metadata_group.attrs[key] = value
        string_dtype = h5py.string_dtype(encoding='utf-8')
        if 'args_json' in metadata_group:
            del metadata_group['args_json']
        metadata_group.create_dataset('args_json', data=metadata['args_json'], dtype=string_dtype)

    def _ensure_static_dataset(self, static_group, name, data):
        if name in static_group:
            return
        static_group.create_dataset(name, data=data)

    def _ensure_dynamic_dataset(self, eval_group, name, row_shape, dtype=np.int64):
        if name in eval_group:
            return eval_group[name]
        return eval_group.create_dataset(name, shape=(0,) + row_shape, maxshape=(None,) + row_shape, dtype=dtype, chunks=True, compression='gzip')

    def _append_dynamic_row(self, dataset, row):
        dataset.resize((dataset.shape[0] + 1,) + dataset.shape[1:])
        dataset[-1] = row

    def ensure_eval_detail_artifact(self, records_generalization):
        if not self.save_eval_detail:
            return None
        self.ensure_result_timestamp()
        self.ensure_eval_detail_static_profiles()
        file_path = self.get_eval_detail_file_path()
        with h5py.File(file_path, 'a') as hf:
            (attrs_group, static_group, eval_group) = self._ensure_root_groups(hf)
            self._ensure_eval_detail_attr_group(attrs_group, records_generalization)
            self.write_metadata_group(hf, DETAIL_KIND, os.path.basename(file_path))
            client_ids = np.array([profile['client_id'] for profile in self.eval_detail_static_profiles], dtype=np.int64)
            class_ids = np.arange(self.args.num_classes, dtype=np.int64)
            train_samples = np.array([profile['train_samples'] for profile in self.eval_detail_static_profiles], dtype=np.int64)
            test_samples = np.array([profile['test_samples'] for profile in self.eval_detail_static_profiles], dtype=np.int64)
            train_label_counts = np.array([profile['train_label_counts'] for profile in self.eval_detail_static_profiles], dtype=np.int64)
            test_label_counts = np.array([profile['test_label_counts'] for profile in self.eval_detail_static_profiles], dtype=np.int64)
            self._ensure_static_dataset(static_group, 'client_ids', client_ids)
            self._ensure_static_dataset(static_group, 'class_ids', class_ids)
            self._ensure_static_dataset(static_group, 'train_samples', train_samples)
            self._ensure_static_dataset(static_group, 'test_samples', test_samples)
            self._ensure_static_dataset(static_group, 'train_label_counts', train_label_counts)
            self._ensure_static_dataset(static_group, 'test_label_counts', test_label_counts)
            if self.global_test_label_counts is not None:
                self._ensure_static_dataset(static_group, 'global_test_label_counts', np.array(self.global_test_label_counts, dtype=np.int64))
            num_clients = len(self.clients)
            num_classes = self.args.num_classes
            self._ensure_dynamic_dataset(eval_group, 'loop_round_idx', tuple(), dtype=np.int64)
            self._ensure_dynamic_dataset(eval_group, 'eval_stage_code', tuple(), dtype=np.int8)
            self._ensure_dynamic_dataset(eval_group, 'personalized_correct', (num_clients,), dtype=np.int64)
            self._ensure_dynamic_dataset(eval_group, 'personalized_total', (num_clients,), dtype=np.int64)
            self._ensure_dynamic_dataset(eval_group, 'personalized_per_class_correct', (num_clients, num_classes), dtype=np.int64)
            self._ensure_dynamic_dataset(eval_group, 'personalized_per_class_total', (num_clients, num_classes), dtype=np.int64)
            if records_generalization:
                self._ensure_dynamic_dataset(eval_group, 'generalization_correct', (num_clients,), dtype=np.int64)
                self._ensure_dynamic_dataset(eval_group, 'generalization_total', (num_clients,), dtype=np.int64)
                self._ensure_dynamic_dataset(eval_group, 'generalization_per_class_correct', (num_clients, num_classes), dtype=np.int64)
                self._ensure_dynamic_dataset(eval_group, 'generalization_per_class_total', (num_clients, num_classes), dtype=np.int64)
        self.saved_eval_detail_path = file_path
        return file_path

    def collect_detailed_personalized_eval_details(self):
        details = []
        for client in self.clients:
            details.append(client.detailed_personalized_test_metrics())
        return details

    def collect_detailed_generalization_eval_details(self):
        if self.global_test_loader is None:
            return None
        details = []
        for client in self.clients:
            if not client.supports_generalization_detail():
                return None
            detail = client.detailed_generalization_test_metrics(self.global_test_loader)
            if detail is None:
                return None
            details.append(detail)
        return details

    def record_eval_detail_snapshot(self, personalized_details, generalization_details=None, round_idx=None, stage_code=None):
        if not self.save_eval_detail or personalized_details is None:
            return
        round_idx_value = self.current_eval_round_idx if round_idx is None else round_idx
        stage_code_value = self.current_eval_stage_code if stage_code is None else stage_code
        if round_idx_value is None:
            round_idx_value = self.infer_eval_round_idx(max(len(self.rs_test_acc) - 1, 0))
        if stage_code_value is None:
            stage_code_value = self.infer_eval_stage_code()
        records_generalization = generalization_details is not None and len(generalization_details) == len(self.clients)
        file_path = self.ensure_eval_detail_artifact(records_generalization=records_generalization)
        if file_path is None:
            return
        personalized_correct = np.array([detail['correct'] for detail in personalized_details], dtype=np.int64)
        personalized_total = np.array([detail['total'] for detail in personalized_details], dtype=np.int64)
        personalized_per_class_correct = np.array([detail['per_class_correct'] for detail in personalized_details], dtype=np.int64)
        personalized_per_class_total = np.array([detail['per_class_total'] for detail in personalized_details], dtype=np.int64)
        with h5py.File(file_path, 'a') as hf:
            eval_group = hf['eval']
            self._append_dynamic_row(eval_group['loop_round_idx'], np.int64(round_idx_value))
            self._append_dynamic_row(eval_group['eval_stage_code'], np.int8(stage_code_value))
            self._append_dynamic_row(eval_group['personalized_correct'], personalized_correct)
            self._append_dynamic_row(eval_group['personalized_total'], personalized_total)
            self._append_dynamic_row(eval_group['personalized_per_class_correct'], personalized_per_class_correct)
            self._append_dynamic_row(eval_group['personalized_per_class_total'], personalized_per_class_total)
            if records_generalization:
                generalization_correct = np.array([detail['correct'] for detail in generalization_details], dtype=np.int64)
                generalization_total = np.array([detail['total'] for detail in generalization_details], dtype=np.int64)
                generalization_per_class_correct = np.array([detail['per_class_correct'] for detail in generalization_details], dtype=np.int64)
                generalization_per_class_total = np.array([detail['per_class_total'] for detail in generalization_details], dtype=np.int64)
                self._append_dynamic_row(eval_group['generalization_correct'], generalization_correct)
                self._append_dynamic_row(eval_group['generalization_total'], generalization_total)
                self._append_dynamic_row(eval_group['generalization_per_class_correct'], generalization_per_class_correct)
                self._append_dynamic_row(eval_group['generalization_per_class_total'], generalization_per_class_total)
        self.clear_eval_context()


    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            (ct, ns, auc) = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        return (ids, num_samples, tot_correct, tot_auc)

    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.clients:
            (cl, ns) = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)
        ids = [c.id for c in self.clients]
        return (ids, num_samples, losses)

    def evaluate(self, round_idx=None, acc=None, loss=None):
        stats_train = self.train_metrics()
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        personalized_details = self.collect_detailed_personalized_eval_details()
        generalization_details = self.collect_detailed_generalization_eval_details()
        metrics = self.compute_metrics(personalized_details, generalization_details)
        metrics['train_loss'] = float(train_loss)
        self.last_metrics = dict(metrics)
        metrics['total_comm_bytes'] = float(self.comm_bytes_total)
        if acc is None:
            self.rs_test_acc.append(metrics['local_test_acc'])
        else:
            acc.append(metrics['local_test_acc'])
        if loss is None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)
        self.append_metric_curves(metrics)
        self.print_metrics_table_from_dict(metrics)
        step = self.resolve_eval_round_idx(round_idx)
        if self.skip_next_detail_record:
            self.skip_next_detail_record = False
            self.clear_eval_context()
            return metrics
        if self.save_eval_detail:
            self.record_eval_detail_snapshot(personalized_details, generalization_details)
        return metrics

    def resolve_eval_round_idx(self, round_idx=None):
        if round_idx is not None:
            return int(round_idx)
        if self.current_eval_round_idx is not None:
            return int(self.current_eval_round_idx)
        return self.infer_eval_round_idx(max(len(self.rs_test_acc) - 1, 0))

    def compute_metrics(self, personalized_details, generalization_details=None):
        self.ensure_eval_detail_static_profiles()
        train_label_counts = np.array([profile['train_label_counts'] for profile in self.eval_detail_static_profiles], dtype=np.int64)
        rare_k = int(getattr(self.args, 'rare_class_k', 2))
        rare_mask = per_client_rare_mask(train_label_counts, rare_k=rare_k)
        local_correct = np.array([detail['correct'] for detail in personalized_details], dtype=np.float64)
        local_total = np.array([detail['total'] for detail in personalized_details], dtype=np.float64)
        local_client_acc = safe_divide(local_correct, local_total)
        local_test_acc = scalar_accuracy(local_correct.sum(), local_total.sum())
        eval_details = generalization_details if generalization_details is not None else personalized_details
        per_client_balanced = []
        global_correct = 0
        global_total = 0
        unseen_correct = 0
        unseen_total = 0
        rare_correct = 0
        rare_total = 0
        per_client_unseen_correct = []
        per_client_unseen_total = []
        per_client_rare_correct = []
        per_client_rare_total = []
        for (client_idx, detail) in enumerate(eval_details):
            per_class_correct = np.asarray(detail['per_class_correct'], dtype=np.float64)
            per_class_total = np.asarray(detail['per_class_total'], dtype=np.float64)
            confusion_matrix = np.asarray(detail['confusion_matrix'], dtype=np.float64)
            global_correct += detail['correct']
            global_total += detail['total']
            per_client_balanced.append(balanced_accuracy(per_class_correct, per_class_total))
            tlc_k = train_label_counts[client_idx]
            unseen_mask = tlc_k == 0
            client_unseen_correct = per_class_correct[unseen_mask].sum()
            client_unseen_total = per_class_total[unseen_mask].sum()
            unseen_correct += client_unseen_correct
            unseen_total += client_unseen_total
            rare_cols = np.flatnonzero(rare_mask[client_idx])
            client_rare_correct = per_class_correct[rare_cols].sum()
            client_rare_total = per_class_total[rare_cols].sum()
            rare_correct += client_rare_correct
            rare_total += client_rare_total
            per_client_unseen_correct.append(client_unseen_correct)
            per_client_unseen_total.append(client_unseen_total)
            per_client_rare_correct.append(client_rare_correct)
            per_client_rare_total.append(client_rare_total)
        per_client_unseen_acc = safe_divide(per_client_unseen_correct, per_client_unseen_total)
        per_client_rare_acc = safe_divide(per_client_rare_correct, per_client_rare_total)
        floor_panel = personalization_floor_panel(local_client_acc, unseen_acc=per_client_unseen_acc, rare_acc=per_client_rare_acc)
        return {'local_test_acc': local_test_acc, 'global_balanced_test_acc': float(np.nanmean(per_client_balanced)) if len(per_client_balanced) else float('nan'), 'client_unseen_class_acc': scalar_accuracy(unseen_correct, unseen_total), 'rare_class_acc': scalar_accuracy(rare_correct, rare_total), 'total_comm_bytes': float(self.comm_bytes_total), 'f25': floor_panel['f25'], 'l25': floor_panel['l25'], 'u25': floor_panel['u25'], 'r25': floor_panel['r25'], 'u25_defined_clients': floor_panel['u25_defined_clients'], 'f25_num_clients': floor_panel['num_clients'], 'f25_tail_size': floor_panel['tail_size']}

    def append_metric_curves(self, metrics):
        self.rs_local_test_acc.append(metrics['local_test_acc'])
        self.rs_global_balanced_test_acc.append(metrics['global_balanced_test_acc'])
        self.rs_client_unseen_class_acc.append(metrics['client_unseen_class_acc'])
        self.rs_rare_class_acc.append(metrics['rare_class_acc'])
        self.rs_total_comm_bytes.append(metrics['total_comm_bytes'])
        self.rs_f25.append(metrics.get('f25', float('nan')))
        self.rs_l25.append(metrics.get('l25', float('nan')))
        self.rs_u25.append(metrics.get('u25', float('nan')))
        self.rs_r25.append(metrics.get('r25', float('nan')))


    def print_metrics_table_from_dict(self, metrics):
        labels = ['Local', 'Global (balanced)', 'Unseen', 'Rare', 'L@25', 'U@25', 'R@25', 'F@25']
        for (label, key) in zip(labels, METRIC_KEYS):
            value = metrics.get(key, float('nan'))
            print(f'{label}: {100 * value:.4f}%' if np.isfinite(value) else f'{label}: N/A')
        print(f"Train loss: {metrics.get('train_loss', float('nan')):.6f}")


    def print_round_banner(self, round_idx):
        print(f"\n{'=' * 20} (Global Round {round_idx}/{self.global_rounds}) {'=' * 20}")

    def print_client_loss_breakdown(self, client_idx, total_selected_clients, client_id, loss_breakdown):
        print(f"(client {client_idx}/{total_selected_clients}) client_id={client_id} loss total={loss_breakdown['total']:.4f} ce={loss_breakdown['ce']:.4f}")
