import datetime
import os
import numpy as np
import torch
from torch import nn
import matplotlib
from tqdm import tqdm
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import itm_eval

class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir    = log_dir
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.log_dir)
        try:
            self.writer     = SummaryWriter(self.log_dir)
            # dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
            # text_input      = ["OK", "OK"]
            # self.writer.add_graph(model, [dummy_input, text_input])
        except:
            pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.cla()
        plt.close("all")

def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model

class EvalCallback():
    def __init__(self, net, gen_val, log_dir, cuda, batch_size=32, eval_flag=True, period=1):
        super(EvalCallback, self).__init__()
        
        self.net                = net
        self.gen_val            = gen_val
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.batch_size         = batch_size
        self.eval_flag          = eval_flag
        self.period             = period

        self.txt_r1 = []
        self.txt_r5 = []
        self.img_r1 = []
        self.img_r5 = []
        self.epoches = []
    
    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = de_parallel(model_eval)
            
            i_features = []
            t_features = []
            for iteration, batch in tqdm(enumerate(self.gen_val)):
                images, texts = batch
                with torch.no_grad():
                    if self.cuda:
                        images  = images.cuda()
                    
                    images_feature = self.net.encode_image(images)
                    i_features.append(images_feature)

            texts       = self.gen_val.dataset.text
            num_text    = len(texts)
            for i in tqdm(range(0, num_text, self.batch_size)):
                text = texts[i: min(num_text, i + self.batch_size)]
                with torch.no_grad():
                    texts_feature = self.net.encode_text(text)
                    t_features.append(texts_feature)

            i_features = torch.cat(i_features, 0)
            t_features = torch.cat(t_features, 0)
            
            i_features  = i_features / i_features.norm(dim=-1, keepdim=True)
            t_features  = t_features / t_features.norm(dim=-1, keepdim=True)

            logits_per_image    = i_features @ t_features.t()
            logits_per_text     = logits_per_image.t()

            logits_per_image    = logits_per_image.cpu().numpy()
            logits_per_text     = logits_per_text.cpu().numpy()
            
            itm_results = itm_eval(logits_per_image, logits_per_text, self.gen_val.dataset.txt2img, self.gen_val.dataset.img2txt)

            self.txt_r1.append(itm_results['txt_r1'])
            self.txt_r5.append(itm_results['txt_r5'])
            self.img_r1.append(itm_results['img_r1'])
            self.img_r5.append(itm_results['img_r5'])
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_R@1_R@5_R@10.txt"), 'a') as f:
                f.write(str(itm_results))
                f.write("\n")
            
            plt.figure()
            plt.plot(self.epoches, self.txt_r1, 'red', linewidth = 2, label='txt_r1')
            plt.plot(self.epoches, self.txt_r5, 'green', linewidth = 2, label='txt_r5')
            plt.plot(self.epoches, self.img_r1, 'blue', linewidth = 2, label='img_r1')
            plt.plot(self.epoches, self.img_r5, 'pink', linewidth = 2, label='img_r5')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Recall')
            plt.title('A Recall Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_recall.png"))
            plt.cla()
            plt.close("all")
            print(itm_results)
            print("Get recall done.")
