import logging
import os
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import tqdm
import torch.optim.lr_scheduler as lr_sched
import math
from lib.config import cfg
import torch.nn.functional as F


logging.getLogger(__name__).addHandler(logging.StreamHandler())
cur_logger = logging.getLogger(__name__)
import numpy as np





def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch = -1,
            setter = set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError("Class '{}' is not a PyTorch nn Module".format(type(model).__name__))

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch = None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))


class CosineWarmupLR(lr_sched._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min = 0, last_epoch = -1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]


def checkpoint_state(model = None, optimizer = None, epoch = None, it = None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return { 'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state }


def save_checkpoint(state, filename = 'checkpoint'):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


def load_checkpoint(model = None, optimizer = None, filename = 'checkpoint', logger = cur_logger):
    if os.path.isfile(filename):
        logger.info("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch'] if 'epoch' in checkpoint.keys() else -1
        it = checkpoint.get('it', 0.0)
        if model is not None and checkpoint['model_state'] is not None:
            model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        logger.info("==> Done")
    else:
        raise FileNotFoundError

    return it, epoch


def load_part_ckpt(model, filename, logger = cur_logger, total_keys = -1):
    if os.path.isfile(filename):
        logger.info("==> Loading part model from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model_state = checkpoint['model_state']

        update_model_state = { key: val for key, val in model_state.items() if key in model.state_dict() }
        state_dict = model.state_dict()
        state_dict.update(update_model_state)
        model.load_state_dict(state_dict)

        update_keys = update_model_state.keys().__len__()
        if update_keys == 0:
            raise RuntimeError
        logger.info("==> Done (loaded %d/%d)" % (update_keys, total_keys))
    else:
        raise FileNotFoundError

def mse_feature_loss(student_feat, teacher_feat):
        student_feat = F.normalize(student_feat, p=2, dim=1)
        teacher_feat = F.normalize(teacher_feat, p=2, dim=1)
        return F.mse_loss(student_feat, teacher_feat)



class Trainer(object):
    def __init__(self, model,teacher_model, model_fn, optimizer, ckpt_dir, lr_scheduler, bnm_scheduler,
                 model_fn_eval, tb_log, eval_frequency = 1, lr_warmup_scheduler = None, warmup_epoch = 2,
                 grad_norm_clip = 1.0,distill_lambda=0.05,total_epochs=30):
                 
        self.model = model  # Student model
        self.teacher_model = teacher_model  # Teacher model
        self.model_fn = model_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.bnm_scheduler = bnm_scheduler
        self.model_fn_eval = model_fn_eval
        self.ckpt_dir = ckpt_dir
        self.eval_frequency = eval_frequency
        self.tb_log = tb_log
        self.lr_warmup_scheduler = lr_warmup_scheduler
        self.warmup_epoch = warmup_epoch
        self.grad_norm_clip = grad_norm_clip
        self.distill_lambda = distill_lambda  # Weight for distillation loss
        self.logger = logging.getLogger() 
        self.current_epoch = 0
        self.distill_warmup = 5         # Number of warmup epochs
        self.use_cosine_anneal = True   # Enable cosine decay of distill_lambda
        self.total_epochs = total_epochs    # Set total epochs here

        # Uncertainty-aware log sigma parameters (learnable)
        self.log_sigma_bce_rpn = torch.nn.Parameter(torch.zeros(1).cuda())
        self.log_sigma_bce_rcnn = torch.nn.Parameter(torch.zeros(1).cuda())
        self.log_sigma_mse_rpn = torch.nn.Parameter(torch.zeros(1).cuda())
        self.log_sigma_mse_rcnn = torch.nn.Parameter(torch.zeros(1).cuda())
        self.log_sigma_backbone = torch.nn.Parameter(torch.zeros(1).cuda())

        # Add to optimizer
        self.optimizer.add_param_group({'params': [
            self.log_sigma_bce_rpn,
            self.log_sigma_bce_rcnn,
            self.log_sigma_mse_rpn,
            self.log_sigma_mse_rcnn,
            self.log_sigma_backbone
        ]})



    

    

    def distillation_loss(self, student_output, teacher_output, temperature=2):
        """
        Compute distillation loss using BCE for classification (since it's binary),
        SmoothL1 for regression (to reduce RPN regression sensitivity), and
        MSE for backbone feature matching.
        """

        # Binary classification distillation using BCEWithLogitsLoss
        bce_loss = nn.BCEWithLogitsLoss()

        # Flatten logits to [N] for BCE
        s_rpn_logits = student_output['rpn_cls'].squeeze(-1)
        t_rpn_probs = torch.sigmoid(teacher_output['rpn_cls'].squeeze(-1))
        bce_rpn_cls = bce_loss(s_rpn_logits, t_rpn_probs)

        s_rcnn_logits = student_output['rcnn_cls'].squeeze(-1)
        t_rcnn_probs = torch.sigmoid(teacher_output['rcnn_cls'].squeeze(-1))
        bce_rcnn_cls = bce_loss(s_rcnn_logits, t_rcnn_probs)

        # SmoothL1 for RPN regression (dense) with clipping
        s_rpn_reg = torch.clamp(student_output['rpn_reg'], -10, 10)
        t_rpn_reg = torch.clamp(teacher_output['rpn_reg'], -10, 10)
        mse_rpn_reg = F.smooth_l1_loss(s_rpn_reg, t_rpn_reg)

        # MSE for RCNN regression (masked)
        reg_valid_mask = student_output['reg_valid_mask'] > 0  # shape: [N]
        valid_count = reg_valid_mask.sum().item()

        if valid_count > 0:
            student_rcnn_reg = torch.clamp(student_output['rcnn_reg'][reg_valid_mask], -10, 10)
            teacher_rcnn_reg = torch.clamp(teacher_output['rcnn_reg'][reg_valid_mask], -10, 10)

            # Optional debug
            print(f"[DEBUG] Valid RCNN regression targets: {valid_count}")
            print(f"[DEBUG] Student RCNN reg mean: {student_rcnn_reg.abs().mean():.4f}")
            print(f"[DEBUG] Teacher RCNN reg mean: {teacher_rcnn_reg.abs().mean():.4f}")

            mse_rcnn_reg = F.mse_loss(student_rcnn_reg, teacher_rcnn_reg)
        else:
            print("[WARN] No valid RCNN regression targets. MSE rcnn_reg set to 0.")
            mse_rcnn_reg = torch.tensor(0.0).to(student_output['rcnn_reg'].device)

        # MSE for backbone features
        mse_backbone = mse_feature_loss(
            student_output['backbone_features'],
            teacher_output['backbone_features']
        )

        # Final total distillation loss
        distill_loss = (
            bce_rpn_cls +
            bce_rcnn_cls +
            0.25 * mse_rpn_reg +    # ↓ Reduced weight
            0.5 * mse_rcnn_reg +
            1.0 * mse_backbone
        )

        # Print/debug
        print(f"BCE rpn_cls: {bce_rpn_cls.item():.4f}")
        print(f"BCE rcnn_cls: {bce_rcnn_cls.item():.4f}")
        print(f"MSE rpn_reg: {mse_rpn_reg.item():.4f}")
        print(f"MSE rcnn_reg: {mse_rcnn_reg.item():.4f}")
        print(f"MSE Backbone: {mse_backbone.item():.4f}")

        # Uncertainty-aware loss combination
        loss_bce_rpn = 0.5 * torch.exp(-2 * self.log_sigma_bce_rpn) * bce_rpn_cls + self.log_sigma_bce_rpn
        loss_bce_rcnn = 0.5 * torch.exp(-2 * self.log_sigma_bce_rcnn) * bce_rcnn_cls + self.log_sigma_bce_rcnn
        loss_mse_rpn = 0.5 * torch.exp(-2 * self.log_sigma_mse_rpn) * mse_rpn_reg + self.log_sigma_mse_rpn
        loss_mse_rcnn = 0.5 * torch.exp(-2 * self.log_sigma_mse_rcnn) * mse_rcnn_reg + self.log_sigma_mse_rcnn
        loss_backbone = 0.5 * torch.exp(-2 * self.log_sigma_backbone) * mse_backbone + self.log_sigma_backbone

        distill_loss = loss_bce_rpn + loss_bce_rcnn + loss_mse_rpn + loss_mse_rcnn + loss_backbone

        return distill_loss, {
            'bce_rpn_cls': bce_rpn_cls.item(),
            'bce_rcnn_cls': bce_rcnn_cls.item(),
            'mse_rpn_reg': mse_rpn_reg.item(),
            'mse_rcnn_reg': mse_rcnn_reg.item(),
            'mse_backbone': mse_backbone.item(),
            'log_sigma_bce_rpn': self.log_sigma_bce_rpn.item(),
            'log_sigma_bce_rcnn': self.log_sigma_bce_rcnn.item(),
            'log_sigma_mse_rpn': self.log_sigma_mse_rpn.item(),
            'log_sigma_mse_rcnn': self.log_sigma_mse_rcnn.item(),
            'log_sigma_backbone': self.log_sigma_backbone.item(),
        }







    
    def _train_it(self, batch):
        self.model.train()
        self.teacher_model.eval()
        self.optimizer.zero_grad()

        loss, tb_dict, disp_dict, ret_dict, teacher_ret_dict = self.model_fn(self.model, self.teacher_model, batch)

        # Warmup check
        if self.current_epoch < self.distill_warmup:
            print(f"[WARMUP] Epoch {self.current_epoch} < {self.distill_warmup} → skipping distillation.")
            total_loss = loss
        else:
            distill_loss, distill_dict = self.distillation_loss(ret_dict, teacher_ret_dict)

            # Optional: visualize teacher vs. student
            if self.current_epoch % 1 == 0 and hasattr(self, "total_it") and self.total_it % 100 == 0:
                from train_utils.viz_utils import plot_cls_heatmap, plot_regression_delta

                os.makedirs('./vis/heatmaps', exist_ok=True)
                os.makedirs('./vis/deltas', exist_ok=True)
                os.makedirs('./vis/logits', exist_ok=True)  # Create directory if it doesn't exist

                # Save raw logits for offline analysis
                np.save(f'./vis/logits/rpn_cls_student_{self.total_it:06d}.npy', ret_dict['rpn_cls'].detach().cpu().numpy())
                np.save(f'./vis/logits/rpn_cls_teacher_{self.total_it:06d}.npy', teacher_ret_dict['rpn_cls'].detach().cpu().numpy())


                plot_cls_heatmap(ret_dict['rpn_cls'], teacher_ret_dict['rpn_cls'], './vis/heatmaps', self.total_it, name='rpn')
                plot_regression_delta(ret_dict['rpn_reg'], teacher_ret_dict['rpn_reg'], './vis/deltas', self.total_it, title='rpn')

                if ret_dict['reg_valid_mask'].sum() > 0:
                    mask = ret_dict['reg_valid_mask'] > 0
                    s_reg = torch.clamp(ret_dict['rcnn_reg'][mask], -10, 10)
                    t_reg = torch.clamp(teacher_ret_dict['rcnn_reg'][mask], -10, 10)
                    plot_regression_delta(s_reg, t_reg, './vis/deltas', self.total_it, title='rcnn')



            # Optional: apply cosine anneal on distill_lambda
            if self.use_cosine_anneal:
                lambda_weight = self.distill_lambda * 0.5 * (1 + math.cos(math.pi * self.current_epoch / self.total_epochs))
            else:
                lambda_weight = self.distill_lambda

            print(f"Epoch: {self.current_epoch}, Loss: {loss.item():.4f}, Distill Loss: {distill_loss.item():.4f}, Lambda: {lambda_weight:.4f}")
            total_loss = loss + lambda_weight * distill_loss
            disp_dict.update(distill_dict)  # Add KD loss terms to log and tbar


        total_loss.backward()
        torch.cuda.empty_cache()
        clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
        self.optimizer.step()

        return total_loss.item(), tb_dict, disp_dict



    def eval_epoch(self, d_loader):
        self.model.eval()

        eval_dict = { }
        total_loss = count = 0.0

        # eval one epoch
        for i, data in tqdm.tqdm(enumerate(d_loader, 0), total = len(d_loader), leave = False, desc = 'val'):
            self.optimizer.zero_grad()

            #loss, tb_dict, disp_dict, teacher_ret_dict = self.model_fn_eval(self.model, self.teacher_model, data)
            loss, tb_dict, disp_dict = self.model_fn_eval(self.model)
            
           

            total_loss += loss.item()
            count += 1
            for k, v in tb_dict.items():
                eval_dict[k] = eval_dict.get(k, 0) + v

        # statistics this epoch
        for k, v in eval_dict.items():
            eval_dict[k] = eval_dict[k] / max(count, 1)

        cur_performance = 0
        if 'recalled_cnt' in eval_dict:
            eval_dict['recall'] = eval_dict['recalled_cnt'] / max(eval_dict['gt_cnt'], 1)
            cur_performance = eval_dict['recall']
        elif 'iou' in eval_dict:
            cur_performance = eval_dict['iou']

        return total_loss / count, eval_dict, cur_performance

    def train(self, start_it, start_epoch, n_epochs, train_loader, test_loader = None, ckpt_save_interval = 5,
              lr_scheduler_each_iter = False):
        eval_frequency = self.eval_frequency if self.eval_frequency > 0 else 1

        it = start_it
        with tqdm.trange(start_epoch, n_epochs, desc = 'epochs') as tbar, \
                tqdm.tqdm(total = len(train_loader), leave = False, desc = 'train') as pbar:
            
            for epoch in tbar:
                self.current_epoch = epoch  # <== Add this

                if self.lr_scheduler is not None and self.warmup_epoch <= epoch and (not lr_scheduler_each_iter):
                    self.lr_scheduler.step(epoch)

                if self.bnm_scheduler is not None:
                    self.bnm_scheduler.step(it)
                    self.tb_log.add_scalar('bn_momentum', self.bnm_scheduler.lmbd(epoch), it)

                # train one epoch
                for cur_it, batch in enumerate(train_loader):
                    if lr_scheduler_each_iter:
                        self.lr_scheduler.step(it)
                        cur_lr = float(self.optimizer.lr)
                        self.tb_log.add_scalar('learning_rate', cur_lr, it)
                    else:
                        if self.lr_warmup_scheduler is not None and epoch < self.warmup_epoch:
                            self.lr_warmup_scheduler.step(it)
                            cur_lr = self.lr_warmup_scheduler.get_lr()[0]
                        else:
                            cur_lr = self.lr_scheduler.get_lr()[0]

                    loss, tb_dict, disp_dict = self._train_it(batch)
                    it += 1
                    self.total_it = it


                    disp_dict.update({ 'loss': loss, 'lr': cur_lr })
                    # print('#################trained_epoch:', epoch)
                    # print('##################n_epochs * cfg.SAVE_MODEL_PREP:', n_epochs * cfg.SAVE_MODEL_PREP)

                    # log to console and tensorboard
                    pbar.update()
                    pbar.set_postfix(dict(total_it = it))
                    tbar.set_postfix(disp_dict)
                    tbar.refresh()
                    self.logger.info(f"[Epoch {epoch}][Iter {it}] " + ', '.join([f"{k}={v:.4f}" for k, v in disp_dict.items()]))

                    self.logger.info("Uncertainty Weights (log_sigma): " +
                        f"RPN_BCE={self.log_sigma_bce_rpn.item():.3f}, " +
                        f"RCNN_BCE={self.log_sigma_bce_rcnn.item():.3f}, " +
                        f"RPN_REG={self.log_sigma_mse_rpn.item():.3f}, " +
                        f"RCNN_REG={self.log_sigma_mse_rcnn.item():.3f}, " +
                        f"Backbone={self.log_sigma_backbone.item():.3f}"
                    )



                    if self.tb_log is not None:
                        self.tb_log.add_scalar('train_loss', loss, it)
                        self.tb_log.add_scalar('learning_rate', cur_lr, it)
                        for key, val in tb_dict.items():
                            self.tb_log.add_scalar('train_' + key, val, it)

                # save trained model
                trained_epoch = epoch + 1
                if (trained_epoch % ckpt_save_interval == 0) and (trained_epoch >= n_epochs * cfg.SAVE_MODEL_PREP):
                    ckpt_name = os.path.join(self.ckpt_dir, 'checkpoint_epoch_%d' % trained_epoch)
                    save_checkpoint(
                            checkpoint_state(self.model, self.optimizer, trained_epoch, it), filename = ckpt_name,
                    )

                # eval one epoch
                if (epoch % eval_frequency) == 0:
                    pbar.close()
                    if test_loader is not None:
                        with torch.set_grad_enabled(False):
                            val_loss, eval_dict, cur_performance = self.eval_epoch(test_loader)

                        if self.tb_log is not None:
                            self.tb_log.add_scalar('val_loss', val_loss, it)
                            for key, val in eval_dict.items():
                                self.tb_log.add_scalar('val_' + key, val, it)

                pbar.close()
                pbar = tqdm.tqdm(total = len(train_loader), leave = False, desc = 'train')
                pbar.set_postfix(dict(total_it = it))

        return None
