from flcore.clients.clientavg import clientAVG

class clientCCVR(clientAVG):

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
