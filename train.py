import torch
from torch import nn, optim
from torchvision import models, datasets, transforms
from collections import OrderedDict

import argparse
import os

from workspace_utils import active_session 

def get_input_args():
    """
    Retrieves command line arguments created and defined using
    the argparse module. get_input_args() returns these
    arguments as an ArgumentParser object.
    Parameters:
     None - run get_input_args() to store command line arguments
    Returns:
     parser.parse_args() - data structure that stores command
     line arguments object
    """
    proj_dir = os.getcwd()
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_directory', type=str, default=(proj_dir + '/flowers'),
                        help='directory containing the /test, /valid, and /train directories')
    
    parser.add_argument('--save_dir', type=str, default=(proj_dir + '/checkpoint'),
                        help='directory to save checkpoint.pth file')
    
    parser.add_argument('--arch', type=str, default='vgg16',
                        help='architecture of pretrained neural network to use (vgg16 or densenet161)')
    
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate to pass through optimizer')
    
    parser.add_argument('--hidden_units', type=int, default=512,
                        help='''number of hidden units to be created in classifier for chosen neural
                                network''')
    
    parser.add_argument('--epochs', type=int, default=3,
                        help='''number of epochs i.e. number of times to pass through training dataset
                                through neural network''')
    
    parser.add_argument('--gpu', action='store_true',
                        help='include \'--gpu\' in command line to use GPU for training')
    
    return parser.parse_args()

def data_loaders(data_dir):
    """
    Performs appropriate transforms on the training, validation and testing images and returns the
    image folders as a dictionary of ImageFolder objects (required for later mapping from class to
    index) and the dataloaders generator to load images.
    Parameters: 
     data_dir - directory containing the /test, /valid and /train folders
    Returns:
     image_datasets - dictionary of folders containing testing, validation, and training images as 
                      ImageFolder objects
     dataloaders - dictionary of generators for testing, validation, and training images as DataLoader
                   objects
    """
    
    #Define transforms for training, validation and testing sets.
    data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30),
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])]),
                       'valid': transforms.Compose([transforms.RandomResizedCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])]),
                       'test': transforms.Compose([transforms.RandomResizedCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])}
    
    #Load datasets with ImageFolder
    image_datasets = {x: datasets.ImageFolder(data_dir + '/{}'.format(x), data_transforms[x]) for x in ['train', 'valid','test']}
    
    #Define dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True) for x in ['train', 'valid', 'test']}
    
    return image_datasets, dataloaders

def vgg16(hidden_units):
    '''
    Loads a pretrained VGG16 model and creates a classifier for
    the model with one hidden layer consisting of a specified
    number of hidden units
    Parameters:
     Hidden units - Number of hidden units for the hidden layer
    Returns:
     Neural network to be used for training with VGG16 architecture
    '''
    model = models.vgg16(pretrained=True)
    #Freezes parameters of the model
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, hidden_units)),
                              ('relu1', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    model.classifier = classifier
    
    return model

def densenet(hidden_units):
    '''
    Loads a pretrained densenet model and creates a classifier
    for the model with one hidden layer consisting of a
    specified number of hidden units
    Parameters:
     Hidden units - Number of hidden units for the hidden layer
    Returns:
     Neural network to be used for training with densenet
     architecture
    '''
    model = models.densenet161(pretrained=True)
    #Freezes parameters of the model
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(2208, hidden_units)),
                              ('relu1', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    model.classifier = classifier
    
    return model

def validation(model, testloader, criterion, device):
    accuracy = 0
    test_loss = 0
    for images, labels in testloader:
        
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy 
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy

def train(epochs, model, dataloaders, optimizer, criterion, device):
    print_every = 1
    steps = 0
    running_loss = 0
    
    with active_session():

        for e in range(epochs):

            # Model in training mode, dropout is on
            model.train()


            print('Epoch: {}'.format(e+1))

            for images, labels in dataloaders['train']:

                images, labels = images.to(device), labels.to(device)
                steps += 1

                optimizer.zero_grad()

                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    # Model in inference mode, dropout is off
                    model.eval()

                        # Turn off gradients for validation, will speed up inference


                    with torch.no_grad():
                        test_loss, accuracy = validation(model, dataloaders['test'], criterion, device)

                        print("Epoch: {}/{}.. ".format(e+1, epochs),
                              "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                              "Test Loss: {:.3f}.. ".format(test_loss/len(dataloaders['test'])),
                              "Test Accuracy: {:.3f}".format(accuracy/len(dataloaders['test'])))

                        running_loss = 0

                        # Make sure dropout and grads are on for training
                        model.train()
    
    return model, optimizer

def save_checkpoint(in_args, model, optimizer, image_datasets):
    checkpoint = {'arch' : in_args.arch,
                  'input_size' : model.classifier.fc1.in_features,
                  'hidden_units' : in_args.hidden_units,
                  'output_size' : model.classifier.fc2.in_features,
                  'optimizer_state_dict' : optimizer.state_dict(),
                  'classifier_state_dict' : model.classifier.state_dict(),
                  'class_to_idx' : image_datasets['train'].class_to_idx}
    torch.save(checkpoint, in_args.save_dir + '/checkpoint.pth')

def main():
    in_args = get_input_args()
    if in_args.gpu:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    image_datasets, dataloaders = data_loaders(in_args.data_directory)
    if in_args.arch == 'vgg16':
        model = vgg16(in_args.hidden_units)
    elif in_args.arch == 'densenet':
        model = densenet(in_args.hidden_units)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_args.learning_rate)
    
    model.to(device)
    
    model, optimizer = train(in_args.epochs, model, dataloaders, optimizer, criterion, device)
    
    print('Training complete')
              
    test_loss, accuracy = validation(model, dataloaders['valid'], criterion, device)
    print("Test Accuracy: {:.3f}".format(accuracy/len(dataloaders['valid'])))
    
    save_checkpoint(in_args, model, optimizer, image_datasets)

if __name__ == '__main__':
    main()