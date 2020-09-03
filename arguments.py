import argparse
def get_inputs():
    parser = argparse.ArgumentParser(description = 'create arguments for user input')
    parser.add_argument('-d','--data_dir', type = str, default = 'flowers', help = 'directory to pull data', required=False, metavar='',action='store')
    parser.add_argument('-s','--save_dir', type = str, default = 'checkpoints', help = 'directory to save checkpoints', required=False, metavar='',action='store')
    parser.add_argument('-p','--checkpoint', type = str, default = 'checkpoint.pth', help = 'save training model for loading later', required=False, metavar='',action='store')
    parser.add_argument('-a','--arch', type = str, default = 'vgg16', help = 'architecture model', required=False, metavar='',action='store')
    parser.add_argument('-l','--learn_rate', type = float, default = 0.001, help = 'learning rate for training', required=False, metavar='',action='store')
    parser.add_argument('-H','--hidden_units', type = int, default = 256, help = 'hidden unit structure', required=False, metavar='',action='store')
    parser.add_argument('-e','--epoch', type = int, default = 9, help = 'number of epoch training', required=False, metavar='',action='store')
    parser.add_argument('-t','--train_steps', type = int, default = 0, help = 'steps', required=False, metavar='',action='store')
    parser.add_argument('-g','--gpu', type = bool, default = True, help = 'GPU device is used to increase the speed, if available', required=False, metavar='',action='store')
    parser.add_argument('-b','--batch_size', type = int, default = 64, help = 'Configure the batch size for training', required=False, metavar='',action='store')
    parser.add_argument('-o','--dropout', type = float, default = 0.5, help = 'set dropout probability', required=False, metavar='',action='store')
    parser.add_argument('-c','--categories', type = str, default = 'cat_to_name.json', help = 'category labels', required=False, metavar='',action='store')
    
    return parser.parse_args()
    