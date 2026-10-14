import os
import subprocess
import time
import torch
from torch.utils.data import DataLoader
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data, read_global_test_data
from utils.output_utils import round_ckpt_dir

class FedAvg(Server):

    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientAVG)
        print(f'\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}')
        print('Finished creating server and clients.')
        self.Budget = []
        self.client_data_profiles = None
        self.rs_global_test_acc = []
        if self.global_test_data is None:
            self.global_test_data = read_global_test_data(self.dataset)
        if self.global_test_data is not None and len(self.global_test_data) > 0 and (self.global_test_loader is None):
            self.global_test_loader = DataLoader(self.global_test_data, batch_size=self.batch_size, drop_last=False, shuffle=False)
            self.global_test_label_counts = self.count_labels(self.global_test_data)

    def train(self):
        completed_round_idx = 0
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            if i % self.eval_gap == 0:
                if i == 0:
                    self.print_initial_setup()
                else:
                    print('Evaluate personalized models')
                    self.set_eval_context(i, 'pre_local_train')
                    self.evaluate(round_idx=i)
            self.print_round_banner(i)
            for (client_idx, client) in enumerate(self.selected_clients, start=1):
                total_selected_clients = len(self.selected_clients)
                print(f'(client {client_idx}/{total_selected_clients}) client_id={client.id} start train_samples={client.train_samples} test_samples={client.test_samples}')
                train_info = client.train()
                train_time = train_info['train_time']
                local_epochs = train_info['local_epochs']
                print(f'(client {client_idx}/{total_selected_clients}) client_id={client.id} done time={train_time:.2f}s local_epochs={local_epochs}')
                self.print_client_loss_breakdown(client_idx, total_selected_clients, client.id, train_info['loss_breakdown'])
            self.receive_models()
            self.aggregate_parameters()
            if getattr(self.args, 'ckpt_every', 0) and (i % self.args.ckpt_every == 0 or i == self.global_rounds):
                ckpt_dir = round_ckpt_dir(self.args)
                torch.save(self.global_model, os.path.join(ckpt_dir, f'round_{i:03d}.pt'))
            round_time = time.time() - s_t
            self.Budget.append(round_time)
            completed_round_idx = i
            print(f'(Global Round {i}/{self.global_rounds}) completed time={round_time:.2f}s')
        self.send_models()
        print('Evaluate final personalized models')
        self.set_eval_context(completed_round_idx, 'post_train_final')
        self.evaluate(round_idx=completed_round_idx)
        self.save_results()
        self.save_global_model()

    def evaluate(self, round_idx=None, acc=None, loss=None):
        return super().evaluate(round_idx=round_idx, acc=acc, loss=loss)

    def print_initial_setup(self):
        print(f"\n{'=' * 20} (Setup Overview) {'=' * 20}")
        self.ensure_client_data_profiles()
        setup_rows = [['Algorithm', self.algorithm], ['Global Rounds', self.global_rounds], ['Eval Gap', self.eval_gap], ['Join Ratio', self.join_ratio], ['Selected Clients / Round', self.current_num_join_clients], ['Configured Local Epochs', self.local_epochs], ['Batch Size', self.batch_size], ['Learning Rate', self.learning_rate], ['Global Generalization Test', 'Enabled' if self.global_test_loader is not None else 'Disabled']]
        self.print_table(['Parameter', 'Value'], setup_rows)
        self.print_hardware_table()
        distribution_rows = []
        for profile in self.client_data_profiles:
            distribution_rows.append([profile['client_id'], profile['train_samples'], profile['test_samples'], self.format_label_distribution(profile['train_label_counts']), self.format_label_distribution(profile['test_label_counts'])])
        self.print_table(['Client ID', 'Train Samples', 'Test Samples', 'Train Distribution', 'Test Distribution'], distribution_rows)
        if self.global_test_label_counts is not None:
            self.print_table(['Global Test Field', 'Value'], [['Samples', len(self.global_test_data)], ['Distribution', self.format_label_distribution(self.global_test_label_counts)]])

    def ensure_client_data_profiles(self):
        if self.client_data_profiles is not None:
            return
        self.client_data_profiles = []
        for client in self.clients:
            train_data = read_client_data(self.dataset, client.id, is_train=True)
            test_data = read_client_data(self.dataset, client.id, is_train=False)
            self.client_data_profiles.append({'client_id': client.id, 'train_samples': client.train_samples, 'test_samples': client.test_samples, 'train_label_counts': self.count_labels(train_data), 'test_label_counts': self.count_labels(test_data)})

    def get_hardware_info_rows(self):
        backend = self.device
        if backend == 'cuda' and torch.cuda.is_available():
            runtime_index = torch.cuda.current_device()
            return [['Backend', 'cuda'], ['Requested Device ID', self.args.device_id], ['Runtime Device Index', runtime_index], ['Hardware Name', torch.cuda.get_device_name(runtime_index)]]
        if backend == 'mps':
            return [['Backend', 'mps'], ['Requested Device ID', 'N/A (ignored by MPS)'], ['Runtime Device Index', 'N/A (single logical device)'], ['Hardware Name', self.get_mps_hardware_name()]]
        return [['Backend', backend], ['Requested Device ID', 'N/A'], ['Runtime Device Index', 'N/A'], ['Hardware Name', 'CPU']]

    def get_mps_hardware_name(self):
        commands = [['system_profiler', 'SPHardwareDataType'], ['system_profiler', 'SPDisplaysDataType']]
        patterns = ['Chip:', 'Model Name:', 'Chipset Model:']
        for command in commands:
            try:
                output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=5)
            except Exception:
                continue
            for line in output.splitlines():
                stripped = line.strip()
                for pattern in patterns:
                    if stripped.startswith(pattern):
                        value = stripped.split(':', 1)[1].strip()
                        if value:
                            return value
        return 'Apple Metal Performance Shaders device'

    def print_hardware_table(self):
        self.print_table(['Hardware Field', 'Value'], self.get_hardware_info_rows())

    def format_accuracy_cell(self, correct, total):
        if total == 0:
            return 'N/A (0/0)'
        return f'{correct / total:.4f} ({correct}/{total})'


