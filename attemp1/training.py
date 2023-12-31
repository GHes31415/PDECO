import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import Dataset, DataLoader

from models import *


# Think of data normalization 

# Define the dataset class
class PDECO_Dataset(Dataset):
    '''
    Dataset = (X,y)
    X = (sensors_1,sensors_2,xt)
    y = solution
    '''
    # Initialize the dataset
    def __init__(self, data):
        self.x = data[0]
        self.y = data[1]
    # Define the getitem function
    def __getitem__(self, index):
        return (self.x[0][index],self.x[1][index],self.x[2]),self.y[index]
    # Define the len function
    def __len__(self):
        return len(self.x[0])


# Define the training loop

def train(model,dataloader,criterion,optimizer):
    model.train()
    running_loss = 0.0

    for inputs,targets in dataloader:
        # Remember to zero grad the optimizer
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs,targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs[0].size(0)

    epoch_loss = running_loss / len(dataloader.dataset)

    return epoch_loss





def network(problem):

    if problem == 'Heat1D':
        branch = [200]
        trunk = [200]

    return branch,trunk


def run():

    return 0






def main(args):
    # Extract the arguments
    problem = args.Problem
    len_control = args.Sc
    len_uncertainty = args.Su
    lr = args.lr
    epochs = args.epochs
    arch_trunk = args.arch_trunk
    arch_branch = args.arch_branch
    dot_layer = args.dot_layer
    eval_point_dim = args.eval_point_dim
    activation_fn = args.activation_fn
    training_data_path = args.training_data_path
    testing_data_path = args.testing_data_path
    data_points = args.data_points
    testing_points= args.testing_points
    eval_point_imag = args.eval_point_imag

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if arch_trunk == [] and arch_branch == []:
        arch_branch,arch_trunk= network(problem)
    else:
        print('Invalid arch_trunk and arch_branch arguments')
        return None

    # Define the neural network
    # branch_control_net,branch_uncertainty_net,trunk_net = assign_networks(len_control,len_uncertainty,eval_point_dim,arch_branch,arch_trunk,dot_layer,activation_fn,device)
    model = MIONET(n_branches=2,input_sizes=[len_control,len_uncertainty,eval_point_dim],architectures=[arch_branch,arch_branch,arch_trunk],output_size=dot_layer,eval_point_imag= eval_point_imag,activation_fn=eval(activation_fn),device=device).to(device)
    # Loss funciton
    criterion = nn.MSELoss()
    # Optimizer, modify to ADAM
    optimizer = optim.SGD(model.parameters(),lr=lr)


    # The data is a dictionary, the keys of the dictionaries are the experiment number 0-num_exps
    training_data = np.load(training_data_path, allow_pickle=True)
    testing_data = np.load(testing_data_path, allow_pickle=True)

    # xt_train = torch.tensor(training_data['xt']).float().to(device)
    # Temporary fix for xt_train

    xt = torch.tensor([(x, y) for x in torch.linspace(0, 1, len_control) for y in torch.linspace(0, 1, len_uncertainty-1)]).to(device)



    sensors_1 = torch.zeros((data_points,len_control)).to(device)
    sensors_2 = torch.zeros((data_points,len_uncertainty)).to(device)
    y_train = torch.zeros((data_points,len(xt))).to(device)

    for i in range(data_points):
        # print(training_data[i]['solution'])
        sensors_1[i,:] = torch.tensor(training_data[i]['sensors'][0])
        sensors_2[i,:] = torch.tensor(training_data[i]['sensors'][1])
        y_train[i,:] = torch.tensor(training_data[i]['solution'])

    X_train = (sensors_1,sensors_2,xt)

    data_train = X_train,y_train
    data_loader = DataLoader(PDECO_Dataset(data_train),batch_size=50,shuffle=True)
    for epoch in range(epochs):
        epoch_loss = train(model,data_loader,criterion,optimizer)
        print('Epoch: {} Loss: {}'.format(epoch,epoch_loss))

    
    # Testing  
    sensors_1 = torch.zeros((testing_points,len_control)).to(device)
    sensors_2 = torch.zeros((testing_points,len_uncertainty)).to(device)
    y_test = torch.zeros((testing_points,len(xt))).to(device)

    for i in range(testing_points):
        sensors_1[i,:] = torch.tensor(testing_data[i]['sensors'][0])
        sensors_2[i,:] = torch.tensor(testing_data[i]['sensors'][1])
        y_test[i,:] = torch.tensor(testing_data[i]['solution'])
    
    X_test = (sensors_1,sensors_2,xt)
    data_test = X_test,y_test
    data_loader_test = DataLoader(PDECO_Dataset(data_test),batch_size=10,shuffle=True)
    model.eval()
    


    





if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Arguments for mionet training")

    # Define command-line arguments
    parser.add_argument("--Problem", type=str,default='Heat1D', help="PDE to solve")
    parser.add_argument("--Sc", type=int, default=11, help="Number of samples control")
    parser.add_argument("--Su", type=int,default=11, help="Number of samples uncertain parameter")
    parser.add_argument("--data_points", type=int, default=100, help="Number of data points")
    parser.add_argument("--testing_points", type=int, default=20, help="Number of testing points")
    parser.add_argument("--eval_point_imag", type=int, default=110, help="(Nx+1)*(Nt+1)")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--epochs", type=int, default= int(1e3), help="Number of training epochs")
    parser.add_argument("--arch_trunk", type=list, default=[], help="Architecture of the trunk")
    parser.add_argument("--arch_branch", type=list, default=[], help="Architecture of the branch")
    parser.add_argument("--dot_layer",type=int,default=200,help="Output of the trunk and branch")
    parser.add_argument("--eval_point_dim", type=int, default=2, help="Dimension of the evaluation point (x,t)")
    parser.add_argument("--activation_fn", type=str, default="nn.Tanh", help="Activation function")
    parser.add_argument("--training_data_path", type=str, default="/work2/Sebas/OUU_MIONET/PDECO/attemp1/data/DR_train.pkl", help="Path to training data")
    parser.add_argument("--testing_data_path", type=str, default="/work2/Sebas/OUU_MIONET/PDECO/attemp1/data/DR_test.pkl", help="Path to test data")   
    # parser.add_argument("--dropout_prob", type=float, default=0.0, help="Dropout probability")
    # parser.add_argument("--seed", type=int, default=0, help="Random seed")
    # parser.add_argument("--logdir", type=str, default="logs", help="Directory for logs")
    # parser.add_argument("--logname", type=str, default="log", help="Name for log file")
    # parser.add_argument("--save", type=bool, default=False, help="Save model")
    # parser.add_argument("--save_dir", type=str, default="saved_models", help="Directory for saved models")
    # parser.add_argument("--save_name", type=str, default="model", help="Name for saved model")
    # parser.add_argument("--load", type=bool, default=False, help="Load model")
    # parser.add_argument("--load_dir", type=str, default="saved_models", help="Directory for loaded models")
    # parser.add_argument("--load_name", type=str, default="model", help="Name for loaded model")
    # parser.add_argument("--test", type=bool, default=False, help="Test model")
    # parser.add_argument("--test_dir", type=str, default="test", help="Directory for test results")
    # parser.add_argument("--test_name", type=str, default="test", help="Name for test results")
    # parser.add_argument("--plot", type=bool, default=False, help="Plot results")
    # parser.add_argument("--plot_dir", type=str, default="plots", help="Directory for plots")
    # parser.add_argument("--plot_name", type=str, default="plot", help="Name for plot")
    # parser.add_argument("--plot_type", type=str, default="png", help="Plot file type")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args)







