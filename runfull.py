import time

import torch
import numpy as np
import argparse
from utils import *
from tensorboardX import SummaryWriter
from SOAP import *
from torch.optim.lr_scheduler import *
import video_readerX as video_reader
from tqdm import tqdm

class Learner:
    def __init__(self):
        self.args = self.parse_command_line()

        self.checkpoint_dir, self.logfile, self.checkpoint_path_validation, self.checkpoint_path_final \
            = get_log_files(self.args.checkpoint_dir, self.args.resume_from_checkpoint, False)

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir)

        current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())

        self.record_dir = os.path.join('runs', current_time)

        self.train_writer = SummaryWriter('{}/{}'.format(self.record_dir, 'Train'))
        self.val_writer = SummaryWriter('{}/{}'.format(self.record_dir, 'Val'))
        
        self.device = torch.device('cuda' if (torch.cuda.is_available() and self.args.num_gpus > 0) else 'cpu')
        self.model = self.init_model()

        self.video_dataset = video_reader.VideoDataset(self.args)
        self.video_loader = torch.utils.data.DataLoader(self.video_dataset, batch_size=1, num_workers=self.args.num_workers)
        self.val_accuracies = TestAccuracies([self.args.dataset])


        self.accuracy_fn = aggregate_accuracy
        
        if self.args.opt == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.opt == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)


        self.scheduler = MultiStepLR(self.optimizer, milestones=self.args.sch, gamma=0.1)
        
        self.start_iteration = 0
        if self.args.resume_from_checkpoint:
            self.load_checkpoint()
        self.optimizer.zero_grad()

    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--dataset", choices=["ssv2", "kinetics", "hmdb", "ucf"], default="kinetics", help="Dataset to use.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate.")
        parser.add_argument("--tasks_per_batch", type=int, default=16, help="Number of tasks between parameter optimizations.")
        parser.add_argument("--checkpoint_dir", "-c", default="./checkpoints/", help="Directory to save checkpoint to.")
        parser.add_argument("--test_model_name", "-m", default="checkpoint_best_val.pt", help="Path to model to load and test.")
        parser.add_argument("--training_iterations", "-i", type=int, default=10000, help="Number of meta-training iterations.")
        parser.add_argument("--resume_from_checkpoint", "-r", dest="resume_from_checkpoint", default=False, action="store_true", help="Restart from latest checkpoint.")
        parser.add_argument("--way", type=int, default=5, help="Way of each task.")
        parser.add_argument("--shot", type=int, default=1, help="Shots per class.")
        parser.add_argument("--query_per_class", "-qpc", type=int, default=5, help="Target samples (i.e. queries) per class used for training.")
        parser.add_argument("--query_per_class_test", "-qpct", type=int, default=1, help="Target samples (i.e. queries) per class used for testing.")
        parser.add_argument('--val_iters', nargs='+', type=int, help='iterations to val at.', default=[100])
        parser.add_argument("--num_val_tasks", type=int, default=100, help="number of random tasks to val on.")
        parser.add_argument("--num_test_tasks", type=int, default=10000, help="number of random tasks to test on.")
        parser.add_argument("--print_freq", type=int, default=1, help="print and log every n iterations.")
        parser.add_argument("--seq_len", type=int, default=8, help="Frames per video.")
        parser.add_argument("--num_workers", type=int, default=8, help="Num dataloader workers.")
        parser.add_argument("--backbone", choices=["resnet18", "resnet34", "resnet50"], default="resnet50", help="backbone")
        parser.add_argument("--opt", choices=["adam", "sgd"], default="sgd", help="Optimizer")
        parser.add_argument("--save_freq", type=int, default=5000, help="Number of iterations between checkpoint saves.")
        parser.add_argument("--img_size", type=int, default=224, help="Input image size to the CNN after cropping.")
        parser.add_argument("--num_gpus", type=int, default=2, help="Number of GPUs to split the ResNet over")
        parser.add_argument('--sch', nargs='+', type=int, help='iters to drop learning rate', default=[2000, 4000, 5000])
        parser.add_argument("--method", choices=["SOAP"], default="SOAP", help="few-shot method to use")
        parser.add_argument("--pretrained_backbone", "-pt", type=str, default=None, help="pretrained backbone path, used by PAL")
        parser.add_argument("--val_on_test", default=False, action="store_true", help="Danger: Validate on the test set, not the validation set. Use for debugging or checking overfitting on test set. Not good practice to use when developing, hyperparameter tuning or training models.")
        parser.add_argument("--debug_loader", default=False, action="store_true", help="Load 1 vid per class for debugging")

        args = parser.parse_args()

        fold_name = "_".join([args.dataset, args.method, args.backbone, args.opt, "{}way".format(args.way), "{}shot".format(args.shot), "segment%d" % args.seq_len, "i{}".format(args.training_iterations)])
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, fold_name)

        args.last_layer_idx = -1

        if args.checkpoint_dir == None:
            print("need to specify a checkpoint dir")
            exit(1)

        if args.backbone == "resnet50":
            args.trans_linear_in_dim = 2048
        else:
            args.trans_linear_in_dim = 512

        if args.dataset == "ssv2":
            args.traintestlist = os.path.join("/home/wenbo/", "Project/Triple-master/splits/ssv2_OTAM")
            args.path = os.path.join("/home/wenbo/", "Dataset/smsm/frames")
        elif args.dataset == "kinetics":
            args.traintestlist = os.path.join("/home/wenbo/", "Project/Triple-master/splits/kinetics_CMN")
            args.path = os.path.join("/home/wenbo/", "Dataset/kinetics/frames")
        elif args.dataset == "ucf":
            args.traintestlist = os.path.join("/home/wenbo/", "Project/Triple-master/splits/ucf_ARN")
            args.path = os.path.join("/home/wenbo/", "Dataset/ucf101/frames")
        elif args.dataset == "hmdb":
            args.traintestlist = os.path.join("/home/wenbo/", "Project/Triple-master/splits/hmdb_ARN")
            args.path = os.path.join("/home/wenbo/", "Dataset/hmdb51/frames")

        return args

    def init_model(self):
        if self.args.method == "SOAP":
            model = SOAP(self.args)

        model = model.to(self.device) 

        if torch.cuda.is_available() and self.args.num_gpus > 1:
            model.distribute_model()
        return model


    def run(self):
        train_accuracies = []
        losses = []
        loss_list = []
        total_iterations = self.args.training_iterations

        iteration = self.start_iteration

        #val_accuraies = [0] * 5
        val_accuraies = []
        test_accuraies = []
        best_val_accuracy = 0

        # for task_dict in tqdm(self.video_loader, ncols=100, total=total_iterations):
        for task_dict in self.video_loader:
            if iteration >= total_iterations:
                break
            iteration += 1
            torch.set_grad_enabled(True)

            task_loss, task_accuracy, align = self.train_task(task_dict)
            train_accuracies.append(task_accuracy)
            losses.append(task_loss)

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            if (iteration + 1) % 1 == 0:
                self.train_writer.add_scalar('Train Loss', task_loss, iteration+1)
                self.train_writer.add_scalar('Train Accuracy', task_accuracy, iteration+1)

            # print training stats
            if (iteration + 1) % self.args.print_freq == 0:
                lr = self.scheduler.get_last_lr()[0]
                print_and_log(self.logfile,
                              'Train Task [{}/{}], Train Loss: {:.7f}, Train Accuracy: {:.7f}, Learning Rate: {:.7f}'
                              .format(iteration, total_iterations, torch.Tensor(losses).mean().item(),
                                      torch.Tensor(train_accuracies).mean().item(), lr))


                if (iteration + 1) % 1 == 0:
                    loss_list.append(torch.Tensor(losses).mean().item())
                # print("loss_list", loss_list)
                train_accuracies = []
                losses = []
            # save checkpoint
            if ((iteration + 1) % self.args.save_freq == 0) and (iteration + 1) != total_iterations:
                self.save_checkpoint(iteration + 1)

            # validate
            # if ((iteration + 1) in self.args.val_iters) and (iteration + 1) != total_iterations:
            if any((iteration + 1) % i == 0 for i in self.args.val_iters) and (iteration + 1) != total_iterations:

                accuracy_dict = self.evaluate("val")
                iter_acc = accuracy_dict[self.args.dataset]["accuracy"]
                val_accuraies.append(iter_acc)

                self.val_accuracies.print(self.logfile, accuracy_dict, mode="val")

                self.val_writer.add_scalar('Val Accuracy', iter_acc, iteration + 1)

                # save checkpoint if best validation score
                if iter_acc > best_val_accuracy:
                    best_val_accuracy = iter_acc
                    self.save_checkpoint(iteration + 1, "checkpoint_best_val.pt")


    def train_task(self, task_dict):
        """
        For one task, runs forward, calculates the loss and accuracy and backprops
        """
        task_dict = self.prepare_task(task_dict)
        model_dict, align = self.model(task_dict['support_set'], task_dict['support_labels'], task_dict['target_set'])
        target_logits = model_dict['logits']

        target_labels = torch.tensor(list(map(int, task_dict['target_labels']))).to(self.device)


        task_loss = self.model.loss(target_logits, target_labels, self.device) / self.args.tasks_per_batch

        task_accuracy = self.accuracy_fn(target_logits, target_labels)

        task_loss.backward(retain_graph=False)

        return task_loss, task_accuracy, align

    def evaluate(self, mode="val"):
        self.model.eval()
        with torch.no_grad():

            self.video_loader.dataset.split = mode
            if mode == "val":
                n_tasks = self.args.num_val_tasks
                mode_name = 'Val'
            elif mode == "test":
                n_tasks = self.args.num_test_tasks
                mode_name = 'Test'

            accuracy_dict ={}
            accuracies = []
            iteration = 0
            item = self.args.dataset
            # for task_dict in tqdm(self.video_loader, ncols=100, total=self.args.num_val_tasks):
            for task_dict in self.video_loader:
                if iteration >= n_tasks:
                    break
                iteration += 1

                task_dict = self.prepare_task(task_dict)
                model_dict, align = self.model(task_dict['support_set'], task_dict['support_labels'], task_dict['target_set'])
                target_logits = model_dict['logits']
                accuracy = self.accuracy_fn(target_logits, task_dict['target_labels'])

                print("{} Task [{}/{}], {} Accuracy: {:.7f}".format(mode_name, iteration, n_tasks, mode_name, accuracy))

                accuracies.append(accuracy.item())
                del target_logits

            accuracy = np.array(accuracies).mean() * 100.0
            # 95% confidence interval
            confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

            accuracy_dict[item] = {"accuracy": accuracy, "confidence": confidence}
            self.video_loader.dataset.split = "train"
        self.model.train()
        
        return accuracy_dict


    def prepare_task(self, task_dict):
        """
        Remove first batch dimension (as we only ever use a batch size of 1) and move data to device.
        """
        for k in task_dict.keys():
            task_dict[k] = task_dict[k][0].to(self.device)
        return task_dict

    def save_checkpoint(self, iteration, name="checkpoint.pt"):
        d = {'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}

        torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint{}.pt'.format(iteration)))   
        torch.save(d, os.path.join(self.checkpoint_dir, name))

    def load_checkpoint(self, name="checkpoint.pt"):
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, name))
        self.start_iteration = checkpoint['iteration']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])


def main():
    learner = Learner()
    learner.run()

if __name__ == "__main__":
    main()