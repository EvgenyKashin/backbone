import torch
import time
import copy
import tqdm


class Bone:
    def __init__(self,
                 model,
                 datasets,
                 criterion,
                 optimizer,
                 metric_fn,
                 batch_size,
                 num_workers,
                 metric_increase=True):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metric_fn = metric_fn
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metric_increase = metric_increase

        self.dataloaders = {  # TODO: не привязываться к train val а делать с теми которые дадут
            'train': torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size,
                                                 shuffle=True, num_workers=num_workers),
            'val': torch.utils.data.DataLoader(datasets['val'], batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)
        }
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def epoch(self, phase):
        running_loss = 0
        running_metric = 0
        for inputs, labels in tqdm.tqdm(self.dataloaders[phase]):  # todo upgrade tqdm
            loss, metric = self.step(inputs, labels, phase)

            running_loss += loss * inputs.size(0)
            running_metric += metric * inputs.size(0)

        running_loss /= len(self.dataloaders[phase].dataset)
        running_metric /= len(self.dataloaders[phase].dataset)

        return running_loss, running_metric

    def step(self, inputs, labels, phase):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                metric = self.metric_fn(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()

            return loss.cpu().data.numpy(), metric.cpu().data.numpy()

    def train(self, num_epochs):
        start_time = time.time()

        # train_acc_history = []
        # val_acc_history = [] # tb
        # best_model_wts = copy.deepcopy(model.state_dict()) # save immediately
        best_metric = None

        def is_better(new_m, old_m):
            if best_metric is None:
                return True
            return new_m > old_m if self.metric_increase else new_m < old_m

        for epoch in range(num_epochs):
            print(f'Epoch: {epoch}/{num_epochs-1}')

            for phase in ['train', 'val']:  # TODO: test
                if phase == 'train':
                    self.model.train()
                    loss, metric = self.epoch(phase)
                else:
                    self.model.eval()
                    loss, metric = self.epoch(phase)

                print(f'{phase} Loss: {loss:.4f}, Metric: {metric:.4f}')

                if phase == 'val' and is_better(metric, best_metric):
                    best_metric = metric
                    best_model_wts = copy.deepcopy(self.model.state_dict())  # TODO: save

                # if phase == 'val': # TODO:TB
                #     val_acc_history.append(epoch_acc)
                # else:
                #     train_acc_history.append(epoch_acc)

            print()

        time_elapsed = time.time() - start_time
        print(f'Training complete in {time_elapsed/60:.0f}m {time_elapsed%60:.0f}s')
        print(f'Best val metric: {best_metric:.4f}')

        # model.load_state_dict(best_model_wts)
