# ***************************************************************************************************************
# Some Masters torch code for the Gaussian kernel (delete the offset alpha as it leads to under-determination)

class gaussianLayer(nn.Module):
    '''
    Family of function used in the internship report, in the form of k(p) = alpha + f(p), with alpha 
    a trainable parameter of the layer and f a gaussian function depending on sigma,mu wich are also 
    trainable parameters of the layer. These parameters are in the tensor 'weights'
    
    inputs : 
    size_in : int, number of pressure levels considered
    levs : list of int, value of the pressure levels considered in Pa.
    '''
    def __init__(self, size_in,levs):
        super(gaussianLayer,self).__init__()
        self.size_in = size_in
        weights = torch.Tensor(3)
        self.weights = nn.Parameter(weights) 
        nn.init.constant_(self.weights, 0.4)
        self.levs = torch.Tensor(levs)/max(levs)

    def forward(self, x):
        kernel = self.weights[0]*torch.ones(self.size_in,1) + torch.unsqueeze(1./(np.sqrt(2.*np.pi)*self.weights[2])*torch.exp(-torch.pow((self.levs - self.weights[1])/self.weights[2], 2.)/2),dim = 1)
        x_int = torch.mm(x, kernel)
        return x_int

class doublegaussianLayer(nn.Module):
    def __init__(self, size_in):
        super(doublegaussianLayer,self).__init__()
        self.size_in = size_in
        weights = torch.Tensor(4)
        self.weights = nn.Parameter(weights) 
        nn.init.constant_(self.weights, 0.5)
        self.p = torch.linspace(0,1,size_in)
        
    def forward(self, x):
        g1 =  torch.unsqueeze(1./(np.sqrt(2.*np.pi)*self.weights[1])*torch.exp(-torch.pow((
            self.p - self.weights[0])/self.weights[1], 2.)/2),dim = 1)
        g2 =  torch.unsqueeze(1./(np.sqrt(2.*np.pi)*self.weights[3])*torch.exp(-torch.pow((
            self.p - self.weights[2])/self.weights[3], 2.)/2),dim = 1)
        kernel = g1 - g2
        x_int = torch.mm(x, kernel)
        return x_int