import torch
from math import exp
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np



class RNevaWrapper(nn.Module):
    
    def __init__(self, downstream_model, criterion, target_function, image_size, foveation_sigma, blur_filter_size, blur_sigma, forgetting, hidden_size, device="cuda"):
        super(RNevaWrapper,self).__init__()
        
        self.image_size = image_size
        self.blur_filter_size = blur_filter_size
        self.blur_sigma = blur_sigma
        self.foveation_sigma = foveation_sigma
        self.forgetting = forgetting
        self.foveation_aggregation = 1
        
        self.internal_representation = None
        self.ones = None
        self.device = device
        
        self.downstream_model = downstream_model
        self.criterion = criterion
        self.target_function = target_function
        
        #inizializziamo la RNN
        self.hidden_size = hidden_size
        self.rnn= nn.RNN(input_size =2,hidden_size=hidden_size,batch_first=True).to(device)
        self.fc = nn.Linear(hidden_size,2).to(device)
        
    def forward(self, x, foveation_positions,h):

        foveation_pos1, h = self.rnn(foveation_positions,h)
        foveation_pos = self.fc(foveation_pos1)
        foveation_pos1 = foveation_pos.transpose(1,2)
        if self.internal_representation is None:
            raise Exception("First set internal representation with function: initialize_scanpath_generation()")
        foveation_area = get_foveation(self.foveation_aggregation, self.foveation_sigma, self.image_size, foveation_pos1)
        current_foveation_area = self.internal_representation + foveation_area
        blurring_mask = torch.clip(self.ones - current_foveation_area, 0, 1)
        applied_blur = self.blur * blurring_mask
        
        output = self.downstream_model(x + applied_blur)
        return output, foveation_pos,h
    
    def initialize_scanpath_generation(self, x, batch_size):
        self.internal_representation = torch.zeros((batch_size, 1, self.image_size, self.image_size), device='cuda')
        self.ones = torch.ones((batch_size, 1, self.image_size, self.image_size), device=self.device)
        self.blur = calculate_blur(x, self.blur_filter_size, self.blur_sigma)
    def create_scanpath(self, x, scanpath_length,foveation_pos=None, h=None):
        batch_size = x.size(0)
        foveation_hist = []
        self.initialize_scanpath_generation(x,batch_size)
        if foveation_pos is None:
            foveation_pos = torch.zeros((batch_size, 2), device=self.device).unsqueeze(1)
        if h is None:
            h = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        for _ in range(scanpath_length):
            output,foveation_pos,h = self(x,foveation_pos,h)
            foveation_hist.append(foveation_pos.squeeze(1))
        return torch.stack(foveation_hist,1),output
    def run_optimization(self,x,labels,optimizer,scanpath_length):
        batch_size = x.size(0)

        targets = target_function(x,labels)
        self.initialize_scanpath_generation(x,batch_size)
        
        scanpath = []
        loss_history = []
        #first hidden state
        h = torch.zeros(1,batch_size,self.hidden_size).to(self.device)
        foveation_pos = torch.zeros((batch_size,2), device=self.device).unsqueeze(1)
        
        for _ in range(scanpath_length):

            output,foveation_pos,h = self(x,foveation_pos,h)#.squeeze(1))
            loss = self.criterion(output, targets)
            total_loss = loss.mean() 
            #azzera gradienti
            self.zero_grad()
            #calcola gradienti
            total_loss.backward(retain_graph=True)
            #aggiorna parametri e pesi RNN
            #optimizer.step()
            for param in self.rnn.parameters():
                param.data -= param.grad * 0.01
            for param in self.fc.parameters():
                param.data -= param.grad * 0.01
            #aggiorna pesi RNN
            #aggiorna stato interno
            current_foveation_mask = get_foveation(1, self.foveation_sigma, self.image_size, foveation_pos.squeeze(1))
            self.internal_representation = (self.internal_representation * self.forgetting + current_foveation_mask).detach()

            scanpath.append(foveation_pos.detach().squeeze(1))
            loss_history.append(loss.detach())
        return torch.stack(scanpath, 1), torch.stack(loss_history,0)  

# Praticamente, noi vorremmo modificare ,
# forward, che gestisce il passaggio in avanti, definendo come calcolare blur e la foveation area
#run optimization il processo di ottimizzazione delle posizioni di foveazione

        
def calc_gaussian(a, std_dev, image_size, positions):
    B = positions.shape[0]
    xa, ya = create_grid(B, image_size)

    xa = xa - positions[:, 0].view(B, 1, 1) #correzione
    ya = ya - positions[:, 1].view(B, 1, 1)
    distance = (xa**2 + ya**2)
    g = a * torch.exp(-distance / std_dev)
    return g.view(B, 1, image_size, image_size)

#function for blur img
def calculate_blur(images, blur_filter_size, sigma=5):
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(window_size, channel, sigma):
        _1D_window = gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    window = create_window(blur_filter_size, 3, sigma).cuda()
    pad = nn.ReflectionPad2d(padding=blur_filter_size // 2)
    imgs_pad = pad(images)
    blured_images = F.conv2d(imgs_pad, window, groups=3)
    blur = blured_images - images
    return blur       
def get_foveation(foveation_aggregation, foveation_sigma, image_size, positions):
    mask = calc_gaussian(foveation_aggregation, foveation_sigma, image_size, positions)
    return mask

#util for calculate gaussian
def create_grid(batch_size, size):
    t = torch.linspace(-1, 1, size).cuda()
    xa, ya = torch.meshgrid([t, t])
    xa = xa.view(1, size, size).repeat(batch_size, 1, 1)
    ya = ya.view(1, size, size).repeat(batch_size, 1, 1)
    return xa, ya
def target_function(x, y): #???
    return y
#function to ca
# Tutto sta nel come noi definiamo il processo di apprendimento del modello. 
# Processo di apprendimento definito nel seguente modo : 
#input : batch di scanpath, in cui ogni scanpath consiste di un'immagine e le relative fissazioni.
# questo modello viene addestrato ad imitare il meccanismo di attenzione visiva umana, ovvero : 
# ripetutamente pone l'attenzione in un "punto" e una zona "fovea" viene messa a fuoco, mentre il resto è sfocato.

#SE VOGLIAMO ADATTARE QUESTO PER PROCESSARE VIDEO : 
#fare modifiche per cambiare input da immagine a sequenza di immagini. usare lstm per predire la zona di foveazione successiva
#in ogni sequenza di immagini, e i pesi vengono aggiornati per ogni sequenza non per ogni frame.
# partiamo sempre dal task : tutto dipende da come è definito il task originale. perché per ogni fotogramma cosa processa?

