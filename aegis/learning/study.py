class Learner:
    def __init__(self, model, params):
        self.model = params['model']
        self.loss_func = params['loss_func']
        self.optimizer = params['optimizer']
        self.lr_scheduler = params['lr_scheduler']

        self.train_dl = params['train_dl']
        self.valid_dl = params['valid_dl']

        self.total_batch = len(params['train_dl'])
        self.num_epochs = params['num_epochs']
        self.device = params['device']

    def start(self):
        for epoch in range(self.num_epochs):
            avg_cost = 0

            for i, data in enumerate(self.train_dl):
                print('batch:', i, '/', self.total_batch)
                X, Y = data[0].to(self.device), data[1].to(self.device)

                self.optimizer.zero_grad()
                hypothesis = self.model(X)
                cost = self.loss_func(hypothesis, Y)
                cost.backward()
                self.optimizer.step()

                avg_cost += cost / self.total_batch

            print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch, avg_cost))        