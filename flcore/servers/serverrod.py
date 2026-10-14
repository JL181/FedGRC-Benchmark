import time
from flcore.clients.clientrod import clientROD
from flcore.servers.serverbase import Server

class FedROD(Server):

    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientROD)
        print(f'\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}')
        print('Finished creating server and clients.')
        self.Budget = []

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            if i % self.eval_gap == 0:
                print(f'\n-------------Round number: {i}-------------')
                print('\nEvaluate personalized models')
                self.set_eval_context(i, 'pre_local_train')
                self.evaluate()
            for client in self.selected_clients:
                client.train()
            self.receive_models()
            self.aggregate_parameters()
            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])
        print('\nBest accuracy.')
        print(max(self.rs_test_acc))
        print('\nAverage time cost per round.')
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        self.save_results()
