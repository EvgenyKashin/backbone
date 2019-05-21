import time
from pathlib import Path
import torch
from . import utils


class Bone:
    def __init__(self,
                 model,
                 datasets,
                 criterion,
                 optimizer,
                 scheduler=None,
                 scheduling_after_ep=True,
                 early_stop_epoch=None,
                 metric_fn=None,
                 metric_increase=False,
                 batch_size=8,
                 num_workers=4,
                 weights_path='weights/best_model.pth'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduling_after_ep = scheduling_after_ep
        self.early_stop_epoch = early_stop_epoch
        self.metric_fn = criterion if metric_fn is None else metric_fn
        self.metric_increase = metric_increase
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.weights_path = Path(weights_path)
        self.epochs_count = 0

        self.dataloaders = {  # TODO: automatically handel all in loop
            'train': torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size,
                                                 shuffle=True, num_workers=num_workers),
            'val': torch.utils.data.DataLoader(datasets['val'], batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)
        }
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.weights_path.parent.mkdir(exist_ok=True)

    def epoch(self, epoch_num, phase):
        running_loss = 0
        running_metric = 0
        pbar = utils.get_pbar(self.dataloaders[phase], f'{phase} {epoch_num + 1 / self.epochs_count}')

        if self.scheduler and not self.scheduling_after_ep:
            self.scheduler.step()

        for inputs, labels in self.dataloaders[phase]:
            loss, metric = self.step(inputs, labels, phase)

            running_loss += loss * inputs.size(0)
            running_metric += metric * inputs.size(0)

            postfix = {'loss': f'{running_loss:.3f}',
                       'metric': f'{running_metric:.3f}'}
            pbar.set_postfix(postfix)
            pbar.update()

        running_loss /= len(self.dataloaders[phase].dataset)
        running_metric /= len(self.dataloaders[phase].dataset)
        pbar.clear()  # TODO: test

        if self.scheduler and self.scheduling_after_ep:
            self.scheduler.step(running_metric)

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

    def fit(self, epochs_count):
        start_time = time.time()
        self.epochs_count = epochs_count
        epoch_without_improvement = 0

        # train_acc_history = []
        # val_acc_history = [] # tb
        # best_model_wts = copy.deepcopy(model.state_dict()) # save immediately
        best_metric = None

        def is_better(new_m, old_m):
            # TODO: add delta
            if best_metric is None:
                return True
            return new_m > old_m if self.metric_increase else new_m < old_m

        for epoch_num in range(epochs_count):
            # print(f'Epoch: {epoch}/{num_epochs-1}')

            for phase in ['train', 'val']:  # TODO: test phase
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                loss, metric = self.epoch(epoch_num, phase)

                print(f'{phase} Loss: {loss:.4f}, Metric: {metric:.4f}')

                if phase == 'val':
                    if is_better(metric, best_metric):
                        best_metric = metric
                        torch.save(self.model.state_dict(), self.weights_path)
                        epoch_without_improvement = 0
                    else:
                        epoch_without_improvement += 1

            if self.early_stop_epoch is not None and epoch_without_improvement == self.early_stop_epoch:
                print('Early stopping')
                break

                # if phase == 'val': # TODO:TB
                #     val_acc_history.append(epoch_acc)
                # else:
                #     train_acc_history.append(epoch_acc)

            # print()

        time_elapsed = time.time() - start_time
        print(f'Training complete in {time_elapsed/60:.0f}m {time_elapsed%60:.0f}s')
        print(f'Best val metric: {best_metric:.4f}')

        # model.load_state_dict(best_model_wts)