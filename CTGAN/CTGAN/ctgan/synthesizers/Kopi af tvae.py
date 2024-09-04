"""TVAESynthesizer module."""

import numpy as np
import pandas as pd
import torch
import zero
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

    def forward(self, input_):
        """Encode the passed `input_`."""
        feature = self.seq(input_)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar, feature


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
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input_):
        """Decode the passed `input_`."""
        return self.seq(input_), self.sigma


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
        name=None
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
        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)
        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        data_dim = self.transformer.output_dimensions
        encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)
        optimizerAE = Adam(
            list(encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr,
            weight_decay=self.l2scale)
        data_iter = iter(loader) 
        print('Training:', self.epochs)
        for i in range(self.epochs):
            try:
                data = next(data_iter)
            except:
                data_iter = iter(loader)
                data = next(data_iter)

            optimizerAE.zero_grad()
            real = data[0].to(self._device)
            mu, std, logvar, feature = encoder(real)
            if ae == False:
                eps = torch.randn_like(std)
                emb = eps * std + mu
            elif ae == True:
                emb = feature
            rec, sigmas = self.decoder(emb)
            loss_1, loss_2 = _loss_function(
                rec, real, sigmas, mu, logvar,
                self.transformer.output_info_list, self.loss_factor
            )
            if ae == False:
                loss = loss_1 + loss_2
            elif ae == True:
                loss = loss_1
            loss.backward()
            optimizerAE.step()
            self.decoder.sigma.data.clamp_(0.01, 1.0)
            if (i + 1) % 1000 == 0:
                print(f"{i + 1}/{self.epochs} {loss}", flush=True)
                
        '''if ae == False:
            mean = torch.zeros(len(train_data), self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self._device)
            return noise
        elif ae == True:
            encoder.eval()
            real = torch.from_numpy(train_data).to(device=self._device, dtype=torch.float32)
            mu, std, logvar, feature = encoder(real)
            return feature'''
            
        encoder.eval()
        real = torch.from_numpy(train_data).to(device=self._device, dtype=torch.float32)
        mu, std, logvar, feature = encoder(real)
        
        if ae == False:
            eps = torch.randn_like(std)
            emb = eps * std + mu
        elif ae == True:
            emb = feature
            
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

        self.decoder.eval()
        
        data, sigmas = self.decoder(latent)
        data = torch.tanh(data)
        data = data.detach().cpu().numpy()
        
        return self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        self.decoder.to(self._device)
