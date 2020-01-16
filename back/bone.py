import time
from pathlib import Path
import shutil
import torch
import torch.nn
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from . import utils


class Bone:
    def __init__(self,
                 model,
                 datasets,
                 criterion,
                 optimizer,
                 scheduler=None,
                 scheduler_after_ep=True,
                 early_stop_epoch=None,
                 metric_fn=None,
                 metric_increase=False,
                 batch_size=8,
                 num_workers=4,
                 resume=False,
                 data_parallel=False,
                 seed=0,
                 weights_path='weights/best_model.pth',
                 log_dir='logs/experiment'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_after_ep = scheduler_after_ep
        self.early_stop_epoch = early_stop_epoch
        self.metric_fn = criterion if metric_fn is None else metric_fn
        self.metric_increase = metric_increase
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resume = resume
        self.data_parallel = data_parallel
        self.seed = seed
        self.weights_path = Path(weights_path)
        self.log_dir = Path(log_dir)
        self.epochs_count = 0
        self.logger = utils.get_logger()

        self.recreate_experiment_folders(from_scratch=False)
        utils.set_seed(seed)

        self.dataloaders = {  # TODO: automatically handel all in loop
            'train': torch.utils.data.DataLoader(datasets['train'],
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=num_workers),
            'val': torch.utils.data.DataLoader(datasets['val'],
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=num_workers)
        }

        if self.resume:
            if not self.weights_path.exists():
                self.logger.error('Resume is not possible, no weights')
            else:
                self.logger.info(f'Resuming from {self.weights_path}')
                checkpoint = torch.load(self.weights_path)
                self.model.load_state_dict(checkpoint)
                # TODO: bug, move model to device

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda' and data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def recreate_experiment_folders(self, from_scratch=False):
        if from_scratch:
            if self.weights_path.parent.exists():
                self.weights_path.unlink()
            if self.log_dir.exists():
                shutil.rmtree(self.log_dir)

        self.weights_path.parent.mkdir(exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.phase_writer = {
            'train': SummaryWriter(self.log_dir / 'train'),
            'val': SummaryWriter(self.log_dir / 'val')
        }

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

    def epoch(self, epoch_num, phase):
        running_loss = 0
        running_metric = 0
        pbar = utils.get_pbar(self.dataloaders[phase],
                              f'{phase} {epoch_num + 1}/{self.epochs_count}')

        if phase == 'val' and self.scheduler and not self.scheduler_after_ep:
            self.scheduler.step()

        for i, (inputs, labels) in enumerate(self.dataloaders[phase]):
            loss, metric = self.step(inputs, labels, phase)

            running_loss += loss * inputs.size(0)
            running_metric += metric * inputs.size(0)
            utils.update_pbar(pbar, loss, metric)

            step = epoch_num * len(self.dataloaders['train']) + i
            self.phase_writer[phase].add_scalar('batch/loss', loss,
                                                global_step=step)
            self.phase_writer[phase].add_scalar('batch/metric', metric,
                                                global_step=step)

        running_loss /= len(self.dataloaders[phase].dataset)
        running_metric /= len(self.dataloaders[phase].dataset)
        utils.update_pbar(pbar, running_loss, running_metric)
        pbar.close()

        self.phase_writer[phase].add_scalar('epoch/loss', running_loss,
                                            global_step=epoch_num)
        self.phase_writer[phase].add_scalar('epoch/metric', running_metric,
                                            global_step=epoch_num)

        if phase == 'val':
            if self.scheduler and self.scheduler_after_ep:
                self.scheduler.step(running_metric)
            lr = utils.get_lr(self.optimizer)
            self.phase_writer[phase].add_scalar('epoch/lr', lr,
                                                global_step=epoch_num)

        return running_loss, running_metric

    def fit(self, epochs_count, from_scratch=False):
        if from_scratch:
            self.recreate_experiment_folders(from_scratch)

        start_time = time.time()
        self.epochs_count = epochs_count
        epoch_without_improvement = 0
        best_metric = None

        def is_better(new_m, old_m, eps=1e-5):
            if old_m is None:
                return True
            return new_m > old_m + eps if self.metric_increase else \
                new_m < old_m - eps

        for epoch_num in range(epochs_count):
            for phase in ['train', 'val']:  # TODO: test phase
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                loss, metric = self.epoch(epoch_num, phase)

                if phase == 'val':
                    if is_better(metric, best_metric):
                        best_metric = metric
                        if self.data_parallel:
                            torch.save(self.model.module.state_dict(),
                                       self.weights_path)
                        else:
                            torch.save(self.model.state_dict(),
                                       self.weights_path)
                        epoch_without_improvement = 0
                        self.logger.debug('Val metric improved')
                    else:
                        epoch_without_improvement += 1
                        self.logger.debug(f'Val metric did not improve for '
                                          f'{epoch_without_improvement} epochs')

            if self.early_stop_epoch is not None and\
                    epoch_without_improvement == self.early_stop_epoch:
                self.logger.info('Early stopping')
                break

        time_elapsed = time.time() - start_time
        self.logger.info(f'Training complete in {time_elapsed/60:.0f}m'
                         f' {time_elapsed%60:.0f}s')
        self.logger.info(f'Best val metric: {best_metric:.4f}')
