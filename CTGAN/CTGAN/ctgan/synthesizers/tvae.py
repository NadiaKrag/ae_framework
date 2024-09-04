"""TVAESynthesizer module."""

import numpy as np
import pandas as pd
import torch
import zero
import opacus
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from ..data_transformer import DataTransformer
from .base import BaseSynthesizer, random_state


class Encoder(Module):
    """Encoder for the TVAESynthesizer.

    Args:
        data_dim (int):
            Dimensions of the data.
        compress_dims (tuple or list of ints):
            Size of each hidden layer.
        embedding_dim (int):
            Size of the output vector.
    """

    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [
                Linear(dim, item),
                ReLU()
            ]
            dim = item

        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, input_, ae=False):
        """Encode the passed `input_`."""
        feature = self.seq(input_)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar, feature

class Encoder_ae(Module):
    """Encoder for the TVAESynthesizer.

    Args:
        data_dim (int):
            Dimensions of the data.
        compress_dims (tuple or list of ints):
            Size of each hidden layer.
        embedding_dim (int):
            Size of the output vector.
    """

    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder_ae, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [
                Linear(dim, item),
                ReLU()
            ]
            dim = item

        self.seq = Sequential(*seq)
        #self.fc1 = Linear(dim, embedding_dim)
        #self.fc2 = Linear(dim, embedding_dim)

    def forward(self, input_, ae=False):
        """Encode the passed `input_`."""
        feature = self.seq(input_)
        return feature

class Decoder(Module):
    """Decoder for the TVAESynthesizer.

    Args:
        embedding_dim (int):
            Size of the input vector.
        decompress_dims (tuple or list of ints):
            Size of each hidden layer.
        data_dim (int):
            Dimensions of the data.
    """

    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        self.sigma = torch.autograd.Variable(torch.ones(data_dim) * 0.1, requires_grad=True) #Parameter(torch.ones(data_dim) * 0.1) #torch.ones(data_dim) * 0.1 torch.ones(data_dim, requires_grad=True) * 0.1 #

    def forward(self, input_):
        """Decode the passed `input_`."""
        return self.seq(input_), self.sigma

class Autoencoder(Module):
    """Autoencoder class that combines Encoder and Decoder."""

    def __init__(
        self,
        data_dim,
        compress_dims,
        embedding_dim,
        decompress_dims,
        ae=False
    ):
        super(Autoencoder, self).__init__()

        #self.encoder = Encoder(data_dim, compress_dims, embedding_dim)
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [
                Linear(dim, item),
                ReLU()
            ]
            dim = item

        self.seq1 = Sequential(*seq)
        if ae == False:
            self.fc1 = Linear(dim, embedding_dim)
            self.fc2 = Linear(dim, embedding_dim)
        
        #self.decoder = Decoder(embedding_dim, decompress_dims, data_dim)
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq2 = Sequential(*seq)
        self.fc3 = torch.autograd.Variable(torch.ones(data_dim) * 0.1, requires_grad=True)

    def forward(self, input_, ae=False):
        """Forward pass through the autoencoder."""
        '''mu, std, logvar, feature = self.encoder(input_)
        if ae:
            emb = feature
        else:
            eps = torch.randn_like(std)
            emb = eps * std + mu
        rec, sigmas = self.decoder(emb)
        return rec, mu, std, logvar, sigmas'''
        feature = self.seq1(input_)
        if ae:
            return self.seq2(feature), self.fc3
        else:
            mu = self.fc1(feature)
            logvar = self.fc2(feature)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            emb = eps * std + mu
            return self.seq2(emb), mu, std, logvar, self.fc3

def _loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor):
    st = 0
    loss = []
    for column_info in output_info:
        for span_info in column_info:
            if span_info.activation_fn != 'softmax':
                ed = st + span_info.dim
                std = sigmas[st]
                eq = x[:, st] - torch.tanh(recon_x[:, st])
                loss.append((eq ** 2 / 2 / (std ** 2)).sum())
                loss.append(torch.log(std) * x.size()[0])
                st = ed

            else:
                ed = st + span_info.dim
                loss.append(cross_entropy(
                    recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
                st = ed
                
    assert st == recon_x.size()[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]

def _loss_function_ae(recon_x, x, sigmas, output_info, factor):
    st = 0
    loss = []
    for column_info in output_info:
        for span_info in column_info:
            if span_info.activation_fn != 'softmax':
                ed = st + span_info.dim
                std = sigmas[st]
                eq = x[:, st] - torch.tanh(recon_x[:, st])
                loss.append((eq ** 2 / 2 / (std ** 2)).sum())
                loss.append(torch.log(std) * x.size()[0])
                st = ed

            else:
                ed = st + span_info.dim
                loss.append(cross_entropy(
                    recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
                st = ed
                
    assert st == recon_x.size()[1]
    #KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return sum(loss) * factor / x.size()[0] #, KLD / x.size()[0]

class TVAESynthesizer(BaseSynthesizer):
    """TVAESynthesizer."""

    def __init__(
        self,
        meta=None,
        embedding_dim=128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        l2scale=1e-5,
        batch_size=500,
        epochs=300,
        lr=1e-3,
        loss_factor=2,
        device="cuda:0",
        name=None,
        private=False,
        epsilon=0.1,
        dp_type=None
    ):
        
        self.__name__ = name
        self.multiprocess = False
        self.datatype = pd.DataFrame
        self.metadata = meta
        
        if self.metadata != None:
            self.num_cols = []
            self.cat_cols = []
            for col in self.metadata['columns'][:-1]:
                if col['type'] == 'Integer' or col['type'] == 'Float':
                    self.num_cols.append(col['name'])
                elif col['type'] == 'Categorical' or col['type'] == 'Ordinal':
                    self.cat_cols.append(col['name'])
        
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims

        self.lr = lr
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.epochs = epochs
        
        self.private = private
        self.epsilon = epsilon
        self.dp_type = dp_type
        
        self._device = torch.device(device)

    @random_state
    def fit(self, train_data, discrete_columns=(), ae=False):
        """Fit the TVAE Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        
        self.transformer = DataTransformer(ae=ae)
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)
        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        data_dim = self.transformer.output_dimensions
        if self.dp_type == 'both':
            self.autoencoder = Autoencoder(data_dim, self.compress_dims, self.embedding_dim, self.decompress_dims, ae=ae).to(self._device)
            self.optimizerAE = Adam(
                list(self.autoencoder.parameters()),
                lr=self.lr,
                weight_decay=self.l2scale)
            if self.private:
                print('PRIVATE TRAINING')
                privacy_engine = opacus.PrivacyEngine(accountant="rdp")
                self.autoencoder, self.optimizerAE, loader = privacy_engine.make_private_with_epsilon(
                    module=self.autoencoder,
                    optimizer=self.optimizerAE,
                    data_loader=loader,
                    target_epsilon = self.epsilon,
                    target_delta = 0.000012,
                    epochs = self.epochs,
                    max_grad_norm = 1.0
                )
        else:
            if ae == False:
                encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
            else:
                encoder = Encoder_ae(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
            self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)
            self.optimizerAE_encoder = Adam(
                list(encoder.parameters()),
                lr=self.lr,
                weight_decay=self.l2scale)
            self.optimizerAE_decoder = Adam(
                list(self.decoder.parameters()),
                lr=self.lr,
                weight_decay=self.l2scale)
        
            if self.private and self.dp_type == 'encoder':
                print('PRIVATE TRAINING ENCODER')
                privacy_engine = opacus.PrivacyEngine(accountant="rdp")
                encoder, self.optimizerAE_encoder, loader = privacy_engine.make_private_with_epsilon(
                    module=encoder,
                    optimizer=self.optimizerAE_encoder,
                    data_loader=loader,
                    target_epsilon = self.epsilon,
                    target_delta = 0.000012,
                    epochs = self.epochs,
                    max_grad_norm = 1.0
                )
                #target_delta not 0.00001 because privacy budget would be too small (regardless of epochs)
            
            if self.private and self.dp_type == 'decoder':
                print('PRIVATE TRAINING DECODER')
                privacy_engine = opacus.PrivacyEngine(accountant="rdp")
                self.decoder, self.optimizerAE_decoder, loader = privacy_engine.make_private_with_epsilon(
                    module=self.decoder,
                    optimizer=self.optimizerAE_decoder,
                    data_loader=loader,
                    target_epsilon = self.epsilon,
                    target_delta = 0.000012,
                    epochs = self.epochs,
                    max_grad_norm = 1.0
                )
                
            if self.private and self.dp_type == 'whole':
                print('PRIVATE TRAINING ENCODER AND DECODER')
                privacy_engine = opacus.PrivacyEngine(accountant="rdp")
                encoder, self.optimizerAE_encoder, loader = privacy_engine.make_private_with_epsilon(
                    module=encoder,
                    optimizer=self.optimizerAE_encoder,
                    data_loader=loader,
                    target_epsilon = self.epsilon,
                    target_delta = 0.000012,
                    epochs = self.epochs,
                    max_grad_norm = 1.0
                )
                privacy_engine = opacus.PrivacyEngine(accountant="rdp")
                self.decoder, self.optimizerAE_decoder, loader = privacy_engine.make_private_with_epsilon(
                    module=self.decoder,
                    optimizer=self.optimizerAE_decoder,
                    data_loader=loader,
                    target_epsilon = self.epsilon,
                    target_delta = 0.000012,
                    epochs = self.epochs,
                    max_grad_norm = 1.0
                )
        
        #print('IS COMPATIBLE', privacy_engine.is_compatible(module=self.decoder, optimizer=self.optimizerAE_decoder, data_loader=loader))
        #print('VALIDATE', privacy_engine.validate(module=self.decoder, optimizer=self.optimizerAE_decoder, data_loader=loader))
        
        data_iter = iter(loader) 
        print('Training:', self.epochs)
        for i in range(self.epochs):
            
            try:
                data = next(data_iter)
            except:
                data_iter = iter(loader)
                data = next(data_iter)
            if self.dp_type == 'both':
                self.optimizerAE.zero_grad()
            else:
                self.optimizerAE_encoder.zero_grad()
                self.optimizerAE_decoder.zero_grad()
            real = data[0].to(self._device)
            if self.dp_type == 'both':
                if ae == False:
                    rec, mu, std, logvar, sigmas = self.autoencoder(real, ae=ae)
                else:
                    rec, sigmas = self.autoencoder(real, ae=ae)
            else:
                if ae == False:
                    mu, std, logvar, feature = encoder(real)
                else:
                    feature = encoder(real)
                if ae == False:
                    eps = torch.randn_like(std)
                    emb = eps * std + mu
                elif ae == True:
                    emb = feature
                rec, sigmas = self.decoder(emb)
            #print('REC', rec)
            #print('SIGMAS', sigmas)
            '''loss_1, loss_2 = _loss_function(
                rec, real, sigmas, mu, logvar,
                self.transformer.output_info_list, self.loss_factor
            )'''
            if ae == False:
                #loss = loss_1 + loss_2
                loss_1, loss_2 = _loss_function(
                rec, real, sigmas, mu, logvar,
                self.transformer.output_info_list, self.loss_factor)
                loss = loss_1 + loss_2
            elif ae == True:
                loss = _loss_function_ae(
                rec, real, sigmas,
                self.transformer.output_info_list, self.loss_factor)
            '''if i == 0:
                for param in encoder.parameters():
                    print('CHECK PARAM GRAD', type(param), param.size(), param.requires_grad, param.grad, param.grad_sample) #, param.grad_sample
                print('CHECK LOSS GRAD', loss, loss.requires_grad, loss.grad)
                print('module BEFORE BACKWARD', type(encoder))
                print('optimizer BEFORE BACKWARD', type(self.optimizerAE_encoder))
                print('data loader BEFORE BACKWARD', type(loader))'''
            #print(loss)
            #for name, param in self.autoencoder.named_parameters():
                #print(name, type(param), param.size(), param.requires_grad, param.grad, param.grad_sample)
            loss.backward()
            if self.dp_type == 'both':
                self.optimizerAE.step()
            else:
                self.optimizerAE_encoder.step()
                self.optimizerAE_decoder.step()
            #self.decoder.sigma.data.clamp_(0.01, 1.0)
            if (i + 1) % 1000 == 0:
                print(f"{i + 1}/{self.epochs} {loss}", flush=True)
        
        if self.dp_type == 'both':
            print('YOU ARE IN BOTH')
            #self.autoencoder.encoder.eval()
            self.autoencoder.eval()
            if ae == False:
                rec, mu, std, logvar, self.sigmas = self.autoencoder(real, ae=ae)
            else:
                rec, self.sigmas = self.autoencoder(real, ae=ae)
            self.autoencoder.seq1.eval()
            real = torch.from_numpy(train_data).to(device=self._device, dtype=torch.float32)
            if ae == False:
                mu, std, logvar, feature = self.autoencoder.encoder(real)
            else:
                feature = self.autoencoder.seq1(real)
            
            if ae == False:
                eps = torch.randn_like(std)
                emb = eps * std + mu
            elif ae == True:
                emb = feature
            return emb
        else:
            encoder.eval()
            real = torch.from_numpy(train_data).to(device=self._device, dtype=torch.float32)
            if ae == False:
                print('CORRECT TVAE')
                mean = torch.zeros(real.shape[0], real.shape[1])
                std = mean + 1
                noise = torch.normal(mean=mean, std=std).to(self._device)
                mu, std, logvar, feature = encoder(noise)
                #mu, std, logvar, feature = encoder(real)
            else:
                feature = encoder(real)
            
            if ae == False:
                eps = torch.randn_like(std)
                emb = eps * std + mu
            elif ae == True:
                emb = feature
            
            if self.private: 
                print('DELETING PRIVATE STUFF')
                del privacy_engine
                del encoder
                del self.optimizerAE_encoder
                del loader
                #####
                #del self.decoder
                #del self.optimizerAE_decoder
                
            return emb

    @random_state
    def sample(self, samples, seed=0):
        """Sample data similar to the training data.

        Args:
            samples (int):
                Number of rows to sample.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        
        #zero.improve_reproducibility(seed)
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)

        self.decoder.eval()
        
        sample_batch_size = 8092
        steps = samples // sample_batch_size + 1
        data = []
        for _ in range(steps):
            mean = torch.zeros(sample_batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self._device)
            #noise can be latent (for AE or for AE-framework) or noise (for VAE)
            if self.dp_type == 'both':
                fake, sigmas = self.autoencoder.decoder(noise)
            else:
                fake, sigmas = self.decoder(noise)
            fake = torch.tanh(fake)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:samples]
        return self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())

    def reconstruction(self, latent, seed=0):
        
        latent = torch.tensor(latent.values).to(self._device, dtype=torch.float32)
        
        #zero.improve_reproducibility(seed)
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)

        if self.dp_type == 'both':
            #self.autoencoder.decoder.eval()
            self.autoencoder.seq2.eval()
            #data, sigmas = self.autoencoder.decoder(latent)
            data = self.autoencoder.seq2(latent)
            sigmas = self.sigmas
        else:
            self.decoder.eval()
            data, sigmas = self.decoder(latent)
            
        data = torch.tanh(data)
        data = data.detach().cpu().numpy()
        
        transform = self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())
        
        return transform

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self.dp_type == 'both':
            self.autoencoder.to(self._device)
        else:
            self.decoder.to(self._device)
        
    def delete_decoder(self):
        del self.decoder
        del self.optimizerAE_decoder
