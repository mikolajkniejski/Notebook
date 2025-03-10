# %%
import torch

# %%

# Let's reconstruct uniform distribution

X = torch.rand((1000)) * 5

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.a = torch.nn.Parameter(torch.randn(1))
        self.b = torch.nn.Parameter(torch.randn(1))

    def forward(self):
        u = torch.rand(1)
        return self.a + (self.b - self.a) * u

n = Net()
for x in X:
    out = n()
    
    loss = (x - out)**2
    loss.backward()
    for i in n.parameters():
        i.data -= i.grad.data * 0.01
        i.grad.zero_() 

# %%

def gumbel(shape=(1), mu=0, b=1):
    assert b > 0
    U = torch.rand(shape)
    return mu - b * torch.log( - torch.log(U))
# %%
class_probs = torch.arange(10, dtype=torch.float16).reshape((2, 5)) .softmax(dim=1)


# %%
torch.nn.functional.one_hot( (class_probs.log() + gumbel(class_probs.shape)).argmax(dim=1)) 


# %%

a = torch.randn((1,512, 28,28)).softmax(dim=1)

# %%
m = torch.nn.AdaptiveMaxPool2d(output_size=(1, 1))

g = 

def gumbel_softmax():
    