import torch
from torch import nn
from torch import optim
from torchvision import models
from collections import OrderedDict
import PIL
import numpy as np
import json
from workspace_utils import active_session


class Model:
    """ Model class """
    
    def __init__(self, train_loader, valid_loader, test_loader, model_pretrained, hidden_units, epochs, learning_rate, save_dir, gpu):
        """ Initialization of the class """
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.model_pretrained = model_pretrained
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.save_dir = save_dir
        self.gpu = gpu
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.input_size = None
        self.cat_to_name = None
        
    def load_pretrained(self, trained=True):
        """ Load a pretrained model and replace the last layer of the model with a desired number of hidden units"""
        if self.model_pretrained=="vgg13":
            self.model = models.vgg13(pretrained=trained)
            self.input_size = 25088
            # classifier
        elif self.model_pretrained=="vgg16":
            self.model = models.vgg16(pretrained=trained)
            self.input_size = 25088
            # classifier
        elif self.model_pretrained=="vgg19":
            self.model = models.vgg19(pretrained=trained)
            self.input_size = 25088
            # classifier
        elif self.model_pretrained=="alexnet":
            self.model = models.alexnet(pretrained=trained) 
            self.input_size = 9216
            # classifier
        else:
            print("invalid model architecture. Please choose among the following options : 'vgg13', 'vgg16', 'vgg19', 'alexnet'.")
            return -1
        # Turn off gradients to not train these layers which are already optimized
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
        classifier = nn.Sequential(OrderedDict([('dropout1', nn.Dropout(p=0.5)),
                                                ('fc1', nn.Linear(self.input_size, self.hidden_units)),
                                                ('relu1', nn.ReLU()),
                                                ('dropout2', nn.Dropout(p=0.5)),
                                                ('fc2', nn.Linear(self.hidden_units, 102)),
                                                ('output',nn.LogSoftmax(dim=1))
                                               ]))
        # Replace the last layer of the pre-trained model
        self.model.classifier = classifier
        
        return self.model

    def train(self):
        """ Train the model using a train dataloader and see the progression using a validation dataloader"""
        # define loss calculation type
        self.criterion = nn.NLLLoss()
        # define optimizer type and allow modifications only on classifier
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.learning_rate)
        # Set up the used device (cuda if available or cpu)
        if self.gpu:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        print("Deviced used for the training: ", device)

        # send model to GPU if available
        self.model.to(device)
    
        # train model (only the replaced layer)
        steps = 0
        running_loss = 0
        print_every = 5
    
        self.model.train()  # set the model to train mode 
        with active_session():  # desactivate the limited time unactive session from udacity

            for epoch in range(self.epochs):
                for train_inputs, train_labels in self.train_loader:
                    steps += 1
            
                    train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)  # move input and label tensors 
                                                                                           # to GPU/CPU device
            
                    self.optimizer.zero_grad()  # reset the gradient so that it is not accumulated
            
                    logps = self.model.forward(train_inputs)  # forward
                    loss = self.criterion(logps, train_labels)  # loss calculation
                    loss.backward() # backward
                    self.optimizer.step()  # step
            
                    running_loss += loss.item()  # track the losses 

                    if steps % print_every == 0:
                        valid_loss = 0
                        valid_accuracy = 0
                        self.model.eval()  # set the model to evaluation mode (no dropout)
                        with torch.no_grad():  # no gradient tracking to speed up the calculation
                            for valid_inputs, valid_labels in self.valid_loader:
                        
                                valid_inputs, valid_labels = valid_inputs.to(device), valid_labels.to(device) # move input and label tensors 
                                                                                                      # to GPU /CPU device
                                logps = self.model.forward(valid_inputs)  # forward
                                batch_loss = self.criterion(logps, valid_labels)  # loss calculation
                        
                                valid_loss += batch_loss.item()  # track the losses 
                        
                                # Calculate accuracy
                                ps = torch.exp(logps)  # probability calculation
                                top_p, top_class = ps.topk(1, dim=1)  # get the top class predicted by the model
                                equals = top_class == valid_labels.view(*top_class.shape)  # check if the prediction is right
                                valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()  # update the accuracy of the model

                        print("Epoch {}/{}.. ".format(epoch + 1, self.epochs),
                              "Train loss: {:.3f}.. ".format(running_loss / print_every),
                              "Valid loss: {:.3f}.. ".format(valid_loss / len(self.valid_loader)),
                              "Valid accuracy: {:.3f}".format(valid_accuracy / len(self.valid_loader)))

                        running_loss = 0  # reset the running losses
                        self.model.train() # set back the model to train for the next batch of train_loader
                
        return self.model
    
    def test(self):
        """Perform validation a the test set"""
        if self.gpu:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        print("Device used for the testing: ", device)
        test_loss = 0
        test_accuracy = 0
        self.model.eval()  # set the model to evaluation mode (no dropout)
        with torch.no_grad():  # no gradient tracking to speed up the calculation
            for test_inputs, test_labels in self.test_loader:
        
                test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

                logps = self.model.forward(test_inputs)
                batch_loss = self.criterion(logps, test_labels)

                test_loss += batch_loss.item()  # track the losses

                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == test_labels.view(*top_class.shape)
                test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        return test_accuracy / len(self.test_loader)
    
    def load_cat_to_name(self, cat_to_name_path):
        """ Load the category to name .json file """
        with open(cat_to_name_path, 'r') as f:
            self.cat_to_name = json.load(f)
        return self.cat_to_name
    
    def include_mapping(self, train_datasets):
        """ Include the mapping of the classes to indices inside the model."""
        self.model.class_to_idx = train_datasets.class_to_idx

    def save(self):
        """ Save the model creating a checkpoint."""
        self.model.to('cpu')
        # creation of the checkpoint dictionnary
        checkpoint = {'model': self.model,
                      'input_size': self.input_size,
                      'hidden_units': self.hidden_units,
                      'ouput_size': 102,
                      'epochs': self.epochs,
                      'optimizer': self.optimizer,
                      'criterion': self.criterion,
                      'class_to_idx': self.model.class_to_idx,
                      'state_dict': self.model.state_dict()
                     }
        # Saving of the checkpoint
        torch.save(checkpoint, self.save_dir + 'checkpoint.pth')
        
    def load(self, checkpoint_path):
        """ Load a model saved in a checkpoint file """
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)  # force all tensors to be on CPU
        self.model = checkpoint['model']
        self.model.to('cpu')
        self.hidden_units = checkpoint['hidden_units']
        self.epochs = checkpoint['epochs']
        self.optimizer = checkpoint['optimizer']
        self.criterion = checkpoint['criterion']
        self.model.class_to_idx = checkpoint['class_to_idx']
        self.model.load_state_dict(checkpoint['state_dict'])
        return self.model
    
    def process_image(self, im_pil):
        """ Process a PIL image for use in a PyTorch model """
        if self.gpu:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
    
        # resize the images where the shortest side is 256 pixels keeping the ratio width/heigth
        im_width, im_height = im_pil.size
        if im_height > im_width:
            # im_width = 256
            new_width = 256
            new_height = int(256 * im_height / im_width)
        else:
            # im_height = 256
            new_width = int(256 * im_width / im_height)
            new_height = 256
        im_pil = im_pil.resize(size=(new_width, new_height), resample=0)
    
        # center crop of the picture to size (224, 224)
        left = (im_width/2 - 224)/2
        top = (im_height/2 - 224)/2
        right = (im_width/2 + 224)/2
        bottom = (im_height/2 + 224)/2
        im_pil = im_pil.crop((left, top, right, bottom))
        
        # convert the pixel value from 0-255 to 0-1
        im_np = np.array(im_pil)
        im_np = im_np / 255
        
        # normalize the images as the network expects (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        im_np = (im_np - mean) /std

        # put the color channel in the 1st dimention of the numpy array.
        im_np = np.transpose(im_np, (2, 0, 1))
    
        return im_np

    def predict(self, image_path, topk=5, class_to_idx=None):
        """ Predict the class (or classes) of an image using the model. """  
        if self.gpu:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')

        # Image preparation
        im_pil = PIL.Image.open(image_path)
        im_processed = self.process_image(im_pil)
        input_data = torch.from_numpy(im_processed)  # convert the image array into a torch tensor
        input_data = input_data.float()  # The default type for weights and biases are torch.FloatTensor 
                                     # so we convert the input_data into torch.FloatTensor
        input_data = input_data.unsqueeze(0) # Adds a dimension of size 1 at the beginning of the tensor to match PyTorch model.
                                         # Normally it has the batch_size at the beggining. 
                                         # In this case, the batch_size = 1 since we only have one picture to treat
                
        # Evaluation
        self.model.eval()  #Put the model in evaluation mode (no dropouts)
        with torch.no_grad():  # no gradient tracking to speed up the calculation
            logps = self.model.forward(input_data)  # prediction of the model
            ps = torch.exp(logps)  # probabilities
            top_p, top_class = ps.topk(topk, dim=1)  # top 5 probabilities and corresponding classes
    
        top_p, top_class = top_p.numpy()[0], top_class.numpy()[0]+1
        
        # if a category to name json file has been loaded
        if self.cat_to_name is not None:
            # replace the category to name in top_class
            top_class = [self.cat_to_name[str(elem)] for elem in top_class]
            
        return top_p, top_class
