import abc
import torch
import os.path as osp

from model.utils import (
    ensure_path,
    Averager, Timer, count_acc,
    compute_confidence_interval,
)
from model.logger import Logger

class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        # ensure_path(
        #     self.args.save_path,
        #     scripts_to_save=['model/models', 'model/networks', __file__],
        # )
        self.logger = Logger(args)

        self.train_step = 0
        self.train_epoch = 0
        self.max_steps = args.episodes_per_epoch * args.max_epoch
        self.dt, self.ft = Averager(), Averager()
        self.bt, self.ot = Averager(), Averager()
        self.timer = Timer()

        # train statistics
        self.trlog = {}
        self.trlog['max_acc'] = 0.0
        self.trlog['max_acc_epoch'] = 0
        self.trlog['max_acc_interval'] = 0.0

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def evaluate(self, data_loader):
        pass
    
    @abc.abstractmethod
    def evaluate_test(self, data_loader):
        pass  
        
    @abc.abstractmethod
    def final_record(self):
        pass    

    def try_evaluate(self, epoch):
        args = self.args
        if self.train_epoch % args.eval_interval == 0:
            eval_ranker = (self.train_epoch + args.eval_interval > self.args.max_epoch)
            vl, va, vap, averagers = self.evaluate(self.val_loader, eval_ranker)
            self.logger.add_scalar('val_loss', float(vl), self.train_epoch)
            self.logger.add_scalar('val_acc', float(va),  self.train_epoch)
            print(f'epoch {epoch}, val, loss={vl:.4f} acc={va:.4f}+{vap:.4f}')

            if averagers["vra"] is not None:
                self.logger.add_scalar('val_ranker_acc', averagers["vra"].item(), self.train_step)
            if averagers["vba"] is not None:
                self.logger.add_scalar('val_best_acc', averagers["vba"].item(), self.train_step)
            if averagers["vra_pos_list"] is not None and type(averagers["vra_pos_list"]) is list:
                for i in range(len(averagers["vra_pos_list"])):
                    self.logger.add_scalar(f'val_ranker_acc_{i}', averagers["vra_pos_list"][i].item(), self.train_step)

            if va >= self.trlog['max_acc']:
                print("Best VAL accuracy!")
                self.trlog['max_acc'] = va
                self.trlog['max_acc_interval'] = vap
                self.trlog['max_acc_epoch'] = self.train_epoch
                self.save_model('max_acc')

    def try_logging(self, averagers):
        args = self.args
        if self.train_step % args.log_interval == 0:
            log_info = 'epoch {}, train {:06g}/{:06g}, total loss={:.4f}'.format(self.train_epoch, self.train_step, self.max_steps, averagers["tl1"].item())
            if averagers["trl"] is not None:
                log_info += ', ranker loss={:.4f}'.format(averagers["trl"].item())
            if averagers["tra"] is not None:
                log_info += ', ranker acc={:.4f}'.format(averagers["tra"].item())
            log_info += ', acc={:.4f}, lr={:.4g}'.format(averagers["ta"].item(), self.optimizer.param_groups[0]['lr'])
            print(log_info)

            self.logger.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.train_step)
            self.logger.add_scalar('train_total_loss', averagers["tl1"].item(), self.train_step)
            self.logger.add_scalar('train_acc',  averagers["ta"].item(), self.train_step)
            if averagers["tat"] is not None:
                self.logger.add_scalar('train_acc_T', averagers["tat"].item(), self.train_step)
            if averagers["trl"] is not None:
                self.logger.add_scalar('train_ranker_loss', averagers["trl"].item(), self.train_step)
            if averagers["tra"] is not None:
                self.logger.add_scalar('train_ranker_acc', averagers["tra"].item(), self.train_step)
            if averagers["tra_pos_list"] is not None and type(averagers["tra_pos_list"]) is list:
                for i in range(len(averagers["tra_pos_list"])):
                    self.logger.add_scalar(f'train_ranker_acc_pos{i}', averagers["tra_pos_list"][i].item(), self.train_step)
            if averagers["trl_list"] is not None and type(averagers["trl_list"]) is list:
                for i in range(len(averagers["trl_list"])):
                    self.logger.add_scalar(f'train_ranker_loss_{i}', averagers["trl_list"][i].item(), self.train_step)
            if averagers["tra_list"] is not None and type(averagers["tra_list"]) is list:
                for i in range(len(averagers["tra_list"])):
                    self.logger.add_scalar(f'train_ranker_acc_{i}', averagers["tra_list"][i].item(), self.train_step)

            print('data_timer: {:.2f} sec, '     \
                  'forward_timer: {:.2f} sec,'   \
                  'backward_timer: {:.2f} sec, ' \
                  'optim_timer: {:.2f} sec'.format(
                        self.dt.item(), self.ft.item(),
                        self.bt.item(), self.ot.item())
                  )
            self.logger.dump()

    def save_model(self, name):
        torch.save(
            dict(params=self.model.state_dict()),
            osp.join(self.args.save_path, name + '.pth')
        )

    def __str__(self):
        return "{}({})".format(
            self.__class__.__name__,
            self.model.__class__.__name__
        )
