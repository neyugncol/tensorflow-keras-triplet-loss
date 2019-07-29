class BaseTrain(object):
    def __init__(self, model, data_loader, config):
        self.model = model
        self.data_loader = data_loader
        self.config = config

    def train(self):
        raise NotImplementedError
