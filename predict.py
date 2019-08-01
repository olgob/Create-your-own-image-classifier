import os
import argparse
from model import Model
import PIL
import process_data

def is_file(file_name):
    """Checks if a file exist"""
    if not os.path.isfile(os.getcwd() + "/" + file_name):
        msg = "{0} does not exist".format(file_name)
        raise argparse.ArgumentTypeError(msg)
    else:
        return file_name

# creation of the parser
parser = argparse.ArgumentParser(description='Description : Predict the name of a flower using a pre-trained model with a picture of the flower as input')

required_args = parser.add_argument_group('required arguments')
required_args.add_argument('--image_path', metavar='\b', required=True, type=is_file, default="flowers/test/1/image_06743.jpg" ,help='path to the image file')
required_args.add_argument('--checkpoint_path', metavar='\b', required=True, type=is_file, default="checkpoint.pth", help='checkpoint of the trained model')

optional_args = parser.add_argument_group('optional arguments')
parser.add_argument('--json_file', metavar='\b', action="store", default=None, type=is_file, help='mapping .json file of categories to real names')
parser.add_argument('--top_k', metavar='\b', action="store", default=1, type=int, help='top k most likely classes')
parser.add_argument('--gpu', action="store_true", help='gpu')

args = parser.parse_args()  # the parameters are now stocked in the namespace args

print("Initialization of model")
model = Model(train_loader=None, valid_loader=None, test_loader=None, model_pretrained=None, hidden_units=None, epochs=None, learning_rate=None, save_dir=None, gpu=args.gpu)

print("Loading parameters of model..")
model.load(os.getcwd() + "/" + args.checkpoint_path)
print("Model_loaded")

print("Prediction by the trained model..")
if args.json_file is not None:
    model.load_cat_to_name(os.getcwd() + "/" + args.json_file)
    
    top_p, top_class = model.predict(image_path=os.getcwd()+"/"+args.image_path, topk=args.top_k) 
    for n in range(len(top_class)):
        print("flower_name: {:30} probability: {:.3f} %".format(top_class[n], top_p[n]*100))
else: 
    print("Prediction by the trained model..")
    top_p, top_class = model.predict(image_path=os.getcwd()+"/"+args.image_path, topk=args.top_k) 
    for n in range(len(top_class)):
        print("flower_cat: {:3} probability: {:.3f} %".format(top_class[n], top_p[n]*100))
