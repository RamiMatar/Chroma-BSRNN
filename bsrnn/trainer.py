import torch
import torch.optim as optim
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_

from data import MusHQDataset, PreprocessedDataset
from model import ChromaModel
from augment import MusicAugmentationDataset
from preprocess import PreprocessValidationSet
import museval
import fast_bss_eval
import tqdm as tq
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp

from hparams import hparams_chroma, hparams_def
import os 
import time


class Trainer():
    def __init__(self, train_loader, validation_loader, chroma_version = 'attention', gpu_id = None, writer_dir = 'runs/', model_save_path = 'best_model.pt', load_from_path = None, hparams = hparams_chroma):
        for key, value in hparams.items():
            setattr(self, key, value)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.train_loader = train_loader
        self.validation_loader = validation_loader
        
        self.best_model_path = model_save_path
        self.gpu_id = gpu_id

        self.train_step = 0
        self.val_step = 0

        self.model = ChromaModel(hparams, chroma_version).to(self.device)
        
        if gpu_id is not None:
            self.device = gpu_id
            torch.cuda.set_device(gpu_id)
            self.model = self.model.to(self.device)
            self.model = DDP(self.model, device_ids=[gpu_id])
        
        self.writer = SummaryWriter(writer_dir)
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.scheduler = StepLR(self.optimizer, step_size=2, gamma=0.995)
        self.loss = torch.nn.L1Loss().to(self.device)
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft = self.n_fft, hop_length = self.hop_length, win_length = self.win_length, power = None).to(self.device)
        self.inverse_spectrogram = torchaudio.transforms.InverseSpectrogram(n_fft = self.n_fft, hop_length = self.hop_length, win_length = self.win_length).to(self.device)
        self.metrics = {
                          'training_losses' : [],
                          'validation_losses': [],
                          'validation_csdrs': [],
                          'training_csdrs': []
                        }

        if load_from_path:
            self.load_trainer(load_from_path)

    def update_training_metrics(self, training_loss, training_csdr, validation_loss = None,  validation_csdr = None):
                self.metrics['training_losses'].append(training_loss)
                self.metrics['training_csdrs'].append(training_csdr)
                if self.device == 0 or self.device == 'cpu' or self.device == 'cuda':
                    if training_loss == min(self.metrics['training_losses']):
                        self.save_model(self.best_model_path[:-3] + '_training_loss.pt')
                    if training_csdr == max(self.metrics['training_csdrs']):
                        self.save_model(self.best_model_path[:-3] + '_training_csdr.pt')
                    self.save_model(self.best_model_path[:-3] + '_latest.pt')

    def update_validation_metrics(self, validation_loss, validation_csdr, training_loss = None, training_csdr = None):
                self.metrics['validation_losses'].append(validation_loss)
                self.metrics['validation_csdrs'].append(validation_csdr)
                if self.device == 0 or self.device == 'cpu' or self.device == 'cuda':
                    if validation_loss == min(self.metrics['validation_losses']):
                        self.save_model(self.best_model_path[:-3] + '_validation_loss.pt')
                    if validation_csdr == min(self.metrics['validation_csdrs']):
                        self.save_model(self.best_model_path[:-3] + '_validation_csdr.pt')

    def train(self, num_epochs, train_to_val = 5):
        with tq.trange(num_epochs, desc="Epochs") as epochs:
            for epoch in epochs:
                with tq.trange(train_to_val) as train_epochs:
                    for train_epoch in train_epochs:
                        training_loss, training_csdr = self.run_train_epoch(epoch)
                        self.update_training_metrics(training_loss, training_csdr)
                validation_loss, validation_csdr = self.run_validation_epoch(epoch)
                self.update_validation_metrics(validation_loss, validation_csdr)
                
                
    def evaluate_and_log(self, step, stft_loss, signal_loss, mixture, reference, estimate, mixture_stft, reference_stft, estimate_stft, subset, compute_sdr = True):
        # reference, estimate shape : (batch_size, Channel, Length)
        batch_size = reference.shape[0]
        assert(reference.shape == estimate.shape and reference.shape == mixture.shape)
        ref_chunks = torch.stack(torch.chunk(reference, reference.shape[2] // self.sample_rate, dim = 2), dim = 1)
        est_chunks = torch.stack(torch.chunk(estimate, estimate.shape[2] // self.sample_rate, dim = 2), dim = 1)
        # ref and est chunks shape : (batch_size, # of 1-second chunks, Channels, Length of 1-second chunks)
        
        try:
            sdr = fast_bss_eval.sdr(est_chunks, ref_chunks, use_cg_iter = 20, clamp_db = 30, load_diag = 2e-5)
            csdr = torch.median(sdr.view(batch_size, -1), dim = 1)[0].mean()
            csdr_loss = -1 * csdr
            csdr = csdr.item()
        except Exception as e:
            print(f"Caught exception: {e}")
            str(torch.isnan(est_chunks).sum())
            str(torch.isnan(ref_chunks).sum())
            csdr = -5
            csdr_loss = torch.tensor(5.0)
            with open('report.txt', 'a') as f:
                # Convert the tensor to a string
                est_str = str(est_chunks)
                ref_str = str(ref_chunks)
                est_sum = str(est_chunks.sum())
                ref_sum = str(ref_chunks.sum())
                ref_std, ref_mean = torch.std_mean(ref_chunks)
                est_std, est_mean = torch.std_mean(est_chunks)
                ref_mean, ref_std, est_mean, est_std = str(ref_mean), str(ref_std), str(est_mean), str(est_std)
                # Write the string to the file
                f.write(f"Ref Mean {ref_mean}\nEst Mean {est_mean}\nRef std {ref_std}\nEst std {est_std} \nEstimate Tensor: {est_str}\nReference Tensor: {ref_str}\n")
        total_loss_val = stft_loss + self.coeff1 * signal_loss + self.coeff2 * csdr_loss.item()
        if step % 25 == 0:    
            tb_mixture, tb_reference, tb_estimate = mixture[0], reference[0], estimate[0]
            if reference.shape[1] > 1:
                tb_reference = torch.mean(tb_reference, dim = 0)
                tb_estimate = torch.mean(tb_estimate, dim = 0)
                tb_mixture = torch.mean(tb_mixture, dim = 0)
    
            self.writer.add_audio(subset + ' Mixture Signal', tb_mixture, step, sample_rate = self.sample_rate)
            self.writer.add_audio(subset + ' Truth Source Signal', tb_reference, step, sample_rate = self.sample_rate)
            self.writer.add_audio(subset + ' Predicted Source Signal', tb_estimate, step, sample_rate = self.sample_rate)

           
            mixture_stft = torch.abs(torch.complex(mixture_stft[:,:,0], mixture_stft[:,:,1]))
            reference_stft = torch.abs(torch.complex(reference_stft[:,:,0], reference_stft[:,:,1]))
            estimate_stft = torch.abs(torch.complex(estimate_stft[:,:,0], estimate_stft[:,:,1]))

            image_grid = torch.stack([mixture_stft.unsqueeze(2), reference_stft.unsqueeze(2), estimate_stft.unsqueeze(2)], dim = 0)
            # log the mixture spectrogram, source spectrogram, and predicted spectrogram
            self.writer.add_images(subset + ' Mixture - Reference - Estimate', image_grid, step, dataformats = "NHWC")
 
            # log the mixture audio, source audio, and predicted audio
       
        # log losses
        self.writer.add_scalar(subset + ' Loss', total_loss_val, step)
        self.writer.add_scalar(subset + ' STFT Loss', stft_loss, step)
        self.writer.add_scalar(subset + ' Signal Loss', signal_loss, step)
        self.writer.add_scalar(subset + ' cSDR Loss', csdr_loss.item(), step)
        
        if csdr is not None:
            self.writer.add_scalar(subset + ' batch cSDR', csdr, step)
        return csdr_loss, csdr

    def evaluate_test_set(self):
        pass

    def eval_track(self, references, estimates, compute_sdr=True):
        if torch.count_nonzero(references) > 100 and torch.count_nonzero(estimates) > 100:
            references = references.transpose(0,1).unsqueeze(0).detach().cpu().numpy()
            estimates = estimates.transpose(0,1).unsqueeze(0).detach().cpu().numpy()
            scores = museval.metrics.bss_eval(
                references, estimates,
                compute_permutation=False,
                framewise_filters=False,
                bsseval_sources_version=False)[:-1]
            sdr = scores[0][0].mean()
            return sdr

    def run_train_epoch(self, epoch_num):
        # set model to train mode and initialize metrics and progress bar
        self.model = self.model.train()
        train_loss = 0
        csdr = 0
        progress_bar = tq.trange(len(self.train_loader), desc="Epoch " + str(epoch_num) + " Progress:")
        for batch_num, data in enumerate(self.train_loader):
            self.train_step += 1
            # Training loop
            # start by zeroing the accumulated gradient.
            self.optimizer.zero_grad()
            # move all tensors received from dataloader to device
            mixtures = data[0].to(self.device)
            sources = data[1].to(self.device)
            
            length = mixtures.shape[-1]
            
            # make prediction from logits
            mask_predictions, mixture_stft = self.model(mixtures)
            stft_predictions = mask_predictions * mixture_stft
            
            stft_truth = self.spectrogram(sources)
            stft_truth = torch.stack([stft_truth.real, stft_truth.imag], dim = 4)

            stft_loss = self.loss(stft_predictions, stft_truth)
            
            stft_predictions_complex = torch.complex(stft_predictions[:,:,:,:,0], stft_predictions[:,:,:,:,1])
            source_predictions = self.inverse_spectrogram(stft_predictions_complex, length)

            signal_loss = self.loss(source_predictions, sources)
            csdr_loss, csdr_val = self.evaluate_and_log(self.train_step, stft_loss.item(), signal_loss.item(), mixtures, 
                        sources, source_predictions, mixture_stft[0,0], stft_truth[0,0],
                        stft_predictions[0,0], 'Training')
            csdr += csdr_val
            total_loss = signal_loss + self.coeff1 * stft_loss + self.coeff2 * csdr_loss
            total_loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=5)
            self.optimizer.step()
            
            
            # update metrics
            train_loss += total_loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix(avg_train_loss = train_loss / (batch_num + 1), avg_cSDR = csdr / (batch_num + 1) )

        return train_loss / len(self.train_loader), csdr / len(self.train_loader)

    def run_validation_epoch(self, epoch_num):
        self.model = self.model.eval()
        validation_loss = 0
        csdr = 0
        progress_bar = tq.trange(len(self.validation_loader), desc="Epoch " + str(epoch_num) + " Progress:")
        with torch.no_grad():
            for batch_num, data in enumerate(self.validation_loader):
                self.val_step += 1
                # move all tensors received from dataloader to device
                mixtures = data[0].to(self.device)
                sources = data[1].to(self.device)
                
                length = mixtures.shape[-1]

                # make prediction from logits
                mask_predictions, mixture_stft = self.model(mixtures)
                stft_predictions = mask_predictions * mixture_stft

                stft_truth = self.spectrogram(sources)
                stft_truth = torch.stack([stft_truth.real, stft_truth.imag], dim = 4)

                stft_loss = self.loss(stft_predictions, stft_truth)
                
                stft_predictions_complex = torch.complex(stft_predictions[:,:,:,:,0], stft_predictions[:,:,:,:,1])
                source_predictions = self.inverse_spectrogram(stft_predictions_complex, length)

                signal_loss = self.loss(source_predictions, sources)
                csdr_loss, csdr_val = self.evaluate_and_log(self.val_step, stft_loss.item(), signal_loss.item(), mixtures, 
                        sources, source_predictions, mixture_stft[0,0], stft_truth[0,0],
                        stft_predictions[0,0], 'Validation ')
                csdr += csdr_val
                total_loss = signal_loss + self.coeff1 * stft_loss + self.coeff2 * csdr_loss
                
                # update metrics
                validation_loss += total_loss.item()
                progress_bar.update(1)
                progress_bar.set_postfix(avg_validation_loss = validation_loss / (batch_num + 1), average_cSDR = csdr / (batch_num + 1))

        return validation_loss / len(self.validation_loader), csdr / len(self.validation_loader)
    
    def run_test_set(self):
        pass
        
    
    def save_model(self, PATH):
        if self.device == 'cuda' or self.device == 'cpu' or self.device == 0:
            if type(self.device) == int: # DDP
                model_dict = self.model.module.state_dict()
            else:
                model_dict = self.model.state_dict()

            checkpoint = {
                'model': model_dict,
                'optim': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'metrics': self.metrics,
                'train_step': self.train_step,
                'val_step': self.val_step
            }
            torch.save(checkpoint, PATH)

    def load_trainer(self, PATH):
        checkpoint = torch.load(PATH)
        if type(self.device) == int:
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])
        self.metrics = checkpoint['metrics']
        self.optimizer.load_state_dict(checkpoint['optim'])
        self.train_step = checkpoint['train_step']
        self.val_step = checkpoint['val_step']
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        
    def report_scores(self):
        for title, metric in self.metrics.items():
            plt.plot(metric)
            plt.title(title)
            plt.show() 

def load_model(PATH, model, optimizer = None):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "11900"
    init_process_group(backend='gloo', rank=rank, world_size=world_size)

def cleanup():
    destroy_process_group()

def prepare_dataloader(dataset, world_size, batch_size, num_workers = 0):
    if world_size <= 1:
        return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers = num_workers
                )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=DistributedSampler(dataset),
        num_workers = num_workers 
    )

def main(rank, world_size, seed, num_workers, epochs, batch_size, epoch_size, validation_per_n_epoch, chroma_version, load_model, save_model, writer_dir):
    print("starting process: ", rank)
    if world_size > 1:
        print("starting ddp")
        ddp_setup(rank, world_size)
        torch.manual_seed(seed)
    else:
        rank = None
    train_dataset = MusicAugmentationDataset('musdb18hq', subset = 'train', split = 'train', epoch_size = epoch_size)
    validation_dataset = PreprocessedDataset('musdb18hq/validation')
    train_loader = prepare_dataloader(train_dataset, world_size, batch_size, num_workers)
    validation_loader = prepare_dataloader(validation_dataset, world_size, batch_size, num_workers)
    trainer = Trainer(train_loader, validation_loader, chroma_version = chroma_version, writer_dir = writer_dir, model_save_path=save_model, load_from_path=load_model, gpu_id = rank)
    trainer.train(epochs, validation_per_n_epoch)
    cleanup()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--epochs', default = 10, type=int, help='Total epochs to train the model')
    parser.add_argument('--seed', default = 42, type=int, help='seed for pseudorandom generation')
    parser.add_argument('--writer_dir', type=str, default = 'tb_logs', help = "directory for tensorboard logs")
    parser.add_argument('--load_model', help='Checkpoint path')
    parser.add_argument('--save_model', default = 'checkpoint.pt', help='Path to save checkpoints, by default checkpoint.pt')
    parser.add_argument('--batch_size', default=4, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--num_workers', default=0, type = int, help='Input number of worker processes for each dataloader')
    parser.add_argument('--validation_per_n_epoch', default = 5, type=int, help='Number of training epochs before doing a validation dataset iteration')
    parser.add_argument('--epoch_size', default = 1000, type = int, help = 'Number of random segments to sample from the datasaet for each epoch, paper uses 10k/epoch')
    parser.add_argument('--chroma_version', default = 'attention', type = str, help = 'Chroma version to use, either attention or fc_group or other')
    args = parser.parse_args() 
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(main, args=(world_size, args.seed, args.num_workers, args.epochs, args.batch_size, args.epoch_size, args.validation_per_n_epoch, args.chroma_version, args.load_model, args.save_model, args.writer_dir), nprocs=world_size)
    else:
        print("hi")
        main(-1, world_size, args.seed, args.num_workers, args.epochs, args.batch_size, args.epoch_size, args.validation_per_n_epoch, args.chroma_version, args.load_model, args.save_model, args.writer_dir)
