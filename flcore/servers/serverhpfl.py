import copy
import numpy as np
import torch
from flcore.servers.serverbase import Server
from flcore.clients.clienthpfl import clientHPFL, fit_head, HPFLPredictor, build_marker, query_manifest, score_predictions
from flcore.heads.extract_features import feats_l4, model_hash
from utils.data_utils import read_client_data
from utils.checkpoint_utils import load_backbone


class HPFL(Server):
    def __init__(self, args, times):
        if not args.backbone_ckpt or args.train_backbone:
            raise ValueError('HPFL requires a frozen --backbone_ckpt')
        if str(args.device).startswith('mps'):
            raise ValueError('HPFL float64 MMD requires CPU or CUDA')
        for name in ('head_epochs', 'head_batch_size', 'feature_batch_size', 'query_batch_size', 'mmd_block_size'):
            if getattr(args, 'hpfl_' + name) < 1:
                raise ValueError(f'hpfl_{name} must be positive')
        if args.hpfl_marker_limit < 0 or args.hpfl_noise_coef < 0 or args.hpfl_head_lr <= 0:
            raise ValueError('Invalid HPFL marker/noise/learning-rate setting')
        if not 0 <= args.hpfl_head_momentum < 1 or args.hpfl_head_weight_decay < 0:
            raise ValueError('Invalid HPFL SGD settings')
        args.global_rounds = 0
        args.local_epochs = args.hpfl_head_epochs
        super().__init__(args, times)
        loaded = load_backbone(args.backbone_ckpt, self.global_model, args)
        if not isinstance(loaded.head, torch.nn.Linear):
            raise ValueError('HPFL requires a full-class linear shared head')
        self.global_model = loaded.to(self.device).eval().requires_grad_(False)
        self.shared_head = copy.deepcopy(loaded.head).cpu()

    def _extract(self, data):
        if data is None or not len(data):
            raise ValueError('HPFL requires nonempty train/test views')
        x, y = feats_l4(self.global_model, data, torch.device(self.device), bs=self.args.hpfl_feature_batch_size)
        if not np.isfinite(x).all() or np.any(y < 0) or np.any(y >= self.num_classes):
            raise ValueError('Invalid HPFL features or labels')
        return torch.from_numpy(x), torch.from_numpy(y)

    def train(self):
        args = self.args
        frozen_hash = model_hash(self.global_model)
        global_features, global_labels = self._extract(self.global_test_data)
        heads, markers = {}, {}
        self.eval_detail_static_profiles = []
        for cid in range(self.num_clients):
            train_x, train_y = self._extract(read_client_data(self.dataset, cid, True))
            test_x, test_y = self._extract(read_client_data(self.dataset, cid, False))
            self.clients.append(clientHPFL(cid, train_x, train_y, test_x, test_y, self.num_classes))
            heads[cid], training = fit_head(self.shared_head, train_x, train_y, args, cid)
            markers[cid] = build_marker(train_x, args.seed, cid, args.hpfl_marker_limit, args.hpfl_noise_coef)
            self.eval_detail_static_profiles.append(dict(client_id=cid, train_samples=len(train_y), test_samples=len(test_y),
                train_label_counts=torch.bincount(train_y, minlength=self.num_classes).tolist(),
                test_label_counts=torch.bincount(test_y, minlength=self.num_classes).tolist()))
            print(f"HPFL client {cid + 1}/{self.num_clients}: {training['optimizer_steps']} SGD steps")
        predictor = HPFLPredictor(self.shared_head, heads, markers, self.device,
                                  args.hpfl_mmd_block_size, args.hpfl_feature_batch_size)
        batches = query_manifest(len(global_features), args.hpfl_query_batch_size, args.hpfl_query_seed, 'global')
        predictions, _ = predictor.predict(global_features, 0, 'selected', batches)
        common_detail = score_predictions(predictions, global_labels.numpy(), self.num_classes)
        for client in self.clients:
            client.predictor, client.state = predictor, 'selected'
            client.local_batches = query_manifest(client.test_samples, args.hpfl_query_batch_size, args.hpfl_query_seed, f'local:{client.id}')
            client.global_detail = common_detail
        self.comm_bytes_total = sum(p.numel() * p.element_size() for head in heads.values() for p in head.parameters())
        self.comm_bytes_total += sum(x.numel() * x.element_size() for x in markers.values())
        self.set_eval_context(0, 'post_train_final')
        self.evaluate(round_idx=0)
        self.save_results()
        if model_hash(self.global_model) != frozen_hash:
            raise RuntimeError('HPFL changed the frozen backbone/shared head')
        self.saved_model_path = self._flat_model_path('deployment')
        torch.save(dict(format='HPFL-deployment-v1', model=copy.deepcopy(self.global_model).cpu(),
                        shared_head=self.shared_head, heads=heads, markers=markers,
                        dataset=self.dataset, seed=args.seed, query_seed=args.hpfl_query_seed,
                        query_batch_size=args.hpfl_query_batch_size,
                        mmd_block_size=args.hpfl_mmd_block_size, feature_batch_size=args.hpfl_feature_batch_size),
                   self.saved_model_path)
