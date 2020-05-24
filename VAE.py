from torch import nn
from torch.autograd import Variable

class VAE(nn.Module):
    def __init__(self, ZDIMS):
        super(VAE, self).__init__()

        # ENCODER
        # 28 x 28 pixels = 784 input pixels, 400 outputs
        self.fc1 = nn.Linear(784, 400)
        # rectified linear unit layer from 400 to 400 max(0, x)
        self.relu = nn.ReLU()
        self.fc21 = nn.Linear(400, ZDIMS)  # mu layer
        self.fc22 = nn.Linear(400, ZDIMS)  # logvariance layer
        # this last layer bottlenecks through ZDIMS connections
        
        # DECODER
        # from bottlenek to hidden 400
        self.fc3 = nn.Linear(ZDIMS, 400)
        # from hidden 400 to 784 outputs
        self.fc4 = nn.Linear(400, 784)
        self.sigmoid = nn.Sigmoid()
        
    def encode(self, x: Variable) -> (Variable, Variable):
        """Input vector x -> fully connected 1 -> ReLU -> (fully connected 21, fully connected 22)
        
        Parameters
        ----------
        x : [128, 784] matrix; 128 digits of 28x28 pixels each
        
        Returns
        -------
        
        (mu, logvar) : ZDIMS mean units one for each latent dimension, ZDIMS variance units one for each latent dimension
        
        """

        # h1 is [128, 400]
        h1 = self.relu(self.fc1(x))  # type: Variable
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu: Variable, logvar: Variable) -> Variable:
        """THE REPARAMETRIZATION IDEA:
        
        Fro each training sample (we get 128 batched as a time)
        
        - take the current learned mu, stddev for each of the ZDIMS deimentions and draw a random sample from that distribution
        - the whole network is trained so that these randomly drawn samples decode to output that looks like the input
        - which will mean that the std, mu will be learned *distributions* that corrctly encode the inputs
        - due to the additional KLD term (see loss_function() below) the disribution will tend to unit Gaussians
        
        Parameters
        ----------
        mu : [128, ZDIMS] mean matrix
        logvar : [128, ZDIMS] variance matrix
        
        Returns
        -------
        
        During training random sample from the learned ZDIMS-dimensional normal distribution; during inference its mean.
        
        """
    
        if self.training:
            # multiply log variance with 0.5, then in-place exponent
            # yielding the standart deviation
            std = logvar.mul(0.5).exp_()  # type: Variable
            # - std.data is the [128, ZDIMS] tensor that is wrapped by std
            # - so eps is [128, ZDIMS] with all elements drawn from a mean 0 and stddev 1 normal disribution that is 128 samples of random ZDIMS-float vectors
            eps = Variable(std.data.new(std.size()).normal_())
            # - sample from a normal distribution with standart deviation = std and mean = mu by multiplying mean 0 stddev 1 sample with desired std and mu
            # - so we have 128 sets (the batch) of random ZDIMS-float vectors sampled from normal disribution with learned std and mu for the current input
            return eps.mul(std).add_(mu)
        else:
            # During inference, we simply spit out the mean of the learned distribution for the current input. We could use a random sample from the distribution, but mu of course has the highest probability.
            return mu
    
    def decode(self, z: Variable) -> Variable:
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))
    
    def forward(self, x: Variable) -> (Variable, Variable, Variable):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar