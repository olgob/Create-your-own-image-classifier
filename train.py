import argparse
import os
from model import Model
import process_data

def is_dir(dir_name):
    """Checks if a path is an actual directory"""
    if not os.path.isdir(dir_name):
        msg = "{0} does not exist".format(dir_name)
        raise argparse.ArgumentTypeError(msg)
    else:
        return dir_name

# creation of the parser
parser = argparse.ArgumentParser(description='Train the feed-foward layer')
# requiered paramters
required_args = parser.add_argument_group('required arguments')
required_args.add_argument('--data_dir', metavar='\b', required=True, type=is_dir, help='directory of the data')
# optional parameters
optional_args = parser.add_argument_group('optional arguments')
parser.add_argument('--save_dir', metavar='\b', action="store", default=os.getcwd()+'/', type=is_dir, help='directory to save checkpoints')
parser.add_argument('--arch', metavar='\b', action="store", default="vgg13", help='architecture of pretrained model [vgg13 vgg16 vgg19 alexnet]')
parser.add_argument('--learning_rate', metavar='\b', action="store", default=0.001, type=float, help='learning_rate')
parser.add_argument('--hidden_units', metavar='\b', action="store", default=2048, type=int, help='hidden units number')
parser.add_argument('--epochs', metavar='\b', action="store", default=2, type=int, help='epochs number')
parser.add_argument('--gpu', action="store_true", help='gpu')
args = parser.parse_args()  # the parameters are now stocked in the namespace args

train_datasets, valid_datasets, test_datasets = process_data.load_datasets(args.data_dir)
print("Datasets loaded.")

train_loader, valid_loader, test_loader = process_data.get_dataloaders(train_datasets, valid_datasets, test_datasets)
print("Dataloaders defined.")

model = Model(train_loader, valid_loader, test_loader, args.arch, args.hidden_units, args.epochs, args.learning_rate, args.save_dir, args.gpu)
print("Download pre-trained model", args.arch, "..")
model.load_pretrained()
print("Pre-trained model loaded")

print("Training of the model starting..")
model.train()
print("Training of the model finished")

print("Test of the accuracy of the model on the test dataset..")
test_accuracy = model.test()
print("Test accuracy: {:.3f}".format(test_accuracy))

print("Saving of the model..")
model.include_mapping(train_datasets)
model.save()
print("Model saved.")
 