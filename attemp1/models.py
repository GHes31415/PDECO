import torch
import torch.nn as nn
import torch.optim as optim

 
class ResBlock(nn.Module):
    def __init__(self,w, activation_fn = None):
        super(ResBlock,self).__init__()
        self.linear = nn.Linear(w,w)
        self.activation_fn = activation_fn
        self.selu = nn.SELU()

    def forward(self,x):
        if self.activation_fn is None:
            return self.selu(self.linear(self.selu(self.linear(x))) + x)
        else:
            return self.activation_fn(self.linear(self.activation_fn(self.linear(x))) + x)  


    
class ResNet(nn.Module):
    def __init__(self,input_size, hidden_sizes ,out_dim , activation_fn  = None,dropout_prob=0.0):
        super(ResNet,self).__init__()
        if out_dim is None:
            out_dim = input_size
        # The * unpacks the list into positional arguments to be evaluated
        self.layers = nn.ModuleList()
        w = hidden_sizes[0]
        # Initial layer allows to include time
        self.layers.append(nn.Linear(input_size, w)),
        if activation_fn is not None:
            self.layers.append(nn.SELU())
        else:
            self.layers.append(activation_fn())
        # Assemble ResNet architecture
        for w in hidden_sizes[1:]:
            self.layers.append(ResBlock(w,activation_fn))
        # Final layer
        self.layers.append(nn.Linear(w,out_dim))
        
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
 


class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation_fn, dropout_prob=0.0):
        super(FeedForwardNN, self).__init__()
        
        self.layers = nn.ModuleList()
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.layers.append(activation_fn)
            if dropout_prob > 0:
                self.layers.append(nn.Dropout(p=dropout_prob))
            prev_size = hidden_size

        self.layers.append(nn.Linear(prev_size, output_size, bias=False))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x




# Define MIONET


class MIONET(nn.Module):

    def __init__(self,n_branches ,input_sizes, architectures, output_size,eval_point_imag, activation_fn, dropout_prob =0.0,device='cpu',model = 'ResNet' ):
        '''
        n_branches = number of branch
        input_sizes = list of input sizes for each branch and trunk
        architectures = list of architectures for each branch and trunk
        output_size = output size of the branch and trunk
        eval_point_imag = number of points to evaluate in the image space
        activation_fn = activation function for the branch and trunk       
        '''
        super(MIONET,self).__init__()

        self.n_branches = n_branches
        self.eval_point_imag = eval_point_imag
        self.device = device
        # nn.ModulesList is a list of nn.Modules
        if model == 'FeedForwardNN':
            # args Feed: input_size, hidden_sizes, output_size, activation_fn, dropout_prob=0.0
            self.branch_nets = nn.ModuleList([FeedForwardNN(input_sizes[i], architectures[i], output_size, activation_fn, dropout_prob).to(device) for i in range(n_branches)])
            
            self.trunk_net = FeedForwardNN(input_sizes[-1], architectures[-1], output_size, activation_fn, dropout_prob)
        elif model == 'ResNet':
            # args ResNet: input_size, hidden_sizes ,out_dim , activation_fn  = None,dropout_prob=0.0
            self.branch_nets = nn.ModuleList([ResNet(input_sizes[i], architectures[i], output_size, activation_fn, dropout_prob).to(device) for i in range(n_branches)])
            
            self.trunk_net = ResNet(input_sizes[-1], architectures[-1], output_size, activation_fn, dropout_prob)

    def forward(self, x):

        # x = (sensor_1,sensor_2,...,sensor_k,xt)
        # sensors are input of branch nets
        # xt is input of trunk net

        sensors = x[:-1]
        xt = x[-1]

        trunk_output = self.trunk_net(xt)
    
        # Initialize the dot product to do element wise product
        dot_product = self.branch_nets[0](sensors[0])

        # Do the element wise product
        for i in range(1,self.n_branches):
            dot_product = torch.mul(dot_product,self.branch_nets[i](sensors[i]))

        # results = torch.zeros((sensors[0].shape[0],self.eval_point_imag)).to(self.device)
        
        # Check einsum torch
        # for i in range(self.eval_point_imag):
        #     results[:, i] = torch.bmm(dot_product.unsqueeze(1), trunk_output[:, i, :].unsqueeze(-1)).squeeze()

        results = torch.einsum('bi,bji->bj', dot_product, trunk_output).to(self.device)
        

        return results#dot_product
