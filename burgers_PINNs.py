import torch 
from collections import OrderedDict #not sure what this does
from pyDOE import lhs #latin-hypercube sampling. 
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
# from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time

np.random.seed(1234)

#CUDA

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# print(device)

# Build NN for solution 
'''
Remember the main structure of a NN is to name the class, 
    inherit the parent torch.nn.module
    remeber to put in the super the name of the NN
    the order of nn is layer then activation function 
    you add them to a list in an orderly fashion 
    then use the command OrderedDict(layer_list)
    then activate it by self.layers = torch.nn.Sequential(layerDict)
    then you define the forward
'''

class DNN(torch.nn.Module):
    def __init__(self,layers):
        super(DNN,self).__init__()

        #parameters
        self.depth = len(layers)-1

        #set up layer order dict

        self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth-1):
            layer_list.append(
                ('layer_%d'%i, torch.nn.Linear(layers[i],layers[i+1]))
            )
            layer_list.append(('activation_%d'%i,self.activation()))
        layer_list.append(
            ('layer_%d'%(self.depth-1),torch.nn.Linear(layers[-2],layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        #deploy layers
        self.layers = torch.nn.Sequential(layerDict)
    
    def forward(self,x):
        out = self.layers(x)
        return out

# Physics-guided NN
''' 
Main function: Compute the derivatives of the NN and plug in
    to the PDE
'''

class PhysicsInformedNN():
    def __init__(self,X_u,u,X_f,layers,lb,ub,nu):
        
        #boundary contidions
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        #data
        self.x_u = torch.tensor(X_u[:,0:1],requires_grad = True).float().to(device)
        self.t_u = torch.tensor(X_u[:,1:2],requires_grad = True).float().to(device)
        self.x_f = torch.tensor(X_f[:,0:1],requires_grad = True).float().to(device)
        self.t_f = torch.tensor(X_f[:,1:2],requires_grad = True).float().to(device)
        self.u = torch.tensor(u).float().to(device)

        self.layers = layers
        self.nu = nu

        #DNN
        self.dnn = DNN(layers).to(device)

        #optimizers
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr = 1.0,
            max_iter = 50000,
            max_eval = 50000,
            history_size = 50,
            tolerance_grad = 1e-5,
            tolerance_change = 1.0*np.finfo(float).eps,
            line_search_fn = "strong_wolfe"
        )

        self.iter = 0

    def net_u(self,x,t):
        u = self.dnn(torch.cat([x,t],dim = 1))
        return u
    def net_f(self,x,t):
        ''' The pytorch autograd version of calculating residuals '''

        u = self.net_u(x,t)

        u_t = torch.autograd.grad(
            u,t,
            grad_outputs = torch.ones_like(u),
            retain_graph = True,
            create_graph = True
        )[0]
        u_x = torch.autograd.grad(
            u,x,
            grad_outputs = torch.ones_like(u),
            retain_graph = True,
            create_graph = True
        )[0]
        u_xx = torch.autograd.grad(
            u_x,x,
            grad_outputs = torch.ones_like(u_x),
            retain_graph = True,
            create_graph = True
        )[0]

        f = u_t+u*u_x -self.nu*u_xx

        return f

    def loss_fn(self):
        
        self.optimizer.zero_grad()

        u_pred = self.net_u(self.x_u,self.t_u)
        f_pred = self.net_f(self.x_f,self.t_f)
        loss_u = torch.mean((self.u-u_pred)**2)
        loss_f = torch.mean(f_pred**2)

        loss = loss_u + loss_f

        loss.backward()
        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e' % (self.iter, loss.item(), loss_u.item(), loss_f.item())
            )
        return loss

    def train(self):
        self.dnn.train()   

        self.optimizer.step(self.loss_fn)     
    
    def predict(self,X):

        x = torch.tensor(X[:,0:1],requires_grad = True).float().to(device)
        t = torch.tensor(X[:,1:2],requires_grad = True).float().to(device)

        self.dnn.eval()
        u = self.net_u(x,t)
        f = self.net_f(x,t)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()

        return u,f



#PDE setup 

nu = 0.01/np.pi

# Number of points in the boundary
N_u = 100
# Number of points for physics loss
N_f = 10_000
layers = [2,20,20,20,20,20,1]

data = scipy.io.loadmat('burgers_shock.mat')

t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = np.real(data['usol']).T

X, T = np.meshgrid(x,t)

X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact.flatten()[:,None]              

# Doman bounds
lb = X_star.min(0)
ub = X_star.max(0)    


# (x,0) partition
xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
uu1 = Exact[0:1,:].T
# (-1,delta t)
xx2 = np.hstack((X[:,0:1], T[:,0:1]))
# Data for left boundary
uu2 = Exact[:,0:1]
# (1, delta t)
xx3 = np.hstack((X[:,-1:], T[:,-1:]))
# data for right boundary
uu3 = Exact[:,-1:]

# Vertically pile up the data
X_u_train = np.vstack([xx1, xx2, xx3])
# Generate N_fx2 samples in [-1,1]x[0,1]
X_f_train = lb + (ub-lb)*lhs(2, N_f)
# Vertically stack the (spatial,temporal) coordinates for training
# in the interior and the boundary data
X_f_train = np.vstack((X_f_train, X_u_train))
# Stack the boundary training data
u_train = np.vstack([uu1, uu2, uu3])

idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
# Randomly select data locations
X_u_train = X_u_train[idx, :]
# Select solution at locations
u_train = u_train[idx,:]

## Training

model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, nu)


               
model.train()

u_pred, f_pred = model.predict(X_star)

error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
print('Error u: %e' % (error_u))                     

U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
Error = np.abs(Exact - U_pred)