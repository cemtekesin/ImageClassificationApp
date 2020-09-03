import argparse

def get_inputs():
    parser = argparse.ArgumentParser(description = 'create arguments for user input to predict')
    parser.add_argument('-l','--load_dir', type = str, default = 'checkpoints', help = 'directory for loading previously saved checkpoints', required=False, metavar='',action='store')
    parser.add_argument('-p','--checkpoint', type = str, default = 'checkpoint.pth', help = 'save training model for loading later', required=False, metavar='',action='store')
    parser.add_argument('-g','--gpu', type = bool, default = True, help = 'GPU device is used to increase the speed, if available', required=False, metavar='',action='store')
    parser.add_argument('-c','--category_names', type = str, default = 'cat_to_name.json', help = 'category labels', required=False, metavar='',action='store')
    parser.add_argument('-t','--top_k', type = int, default = 3, help = 'returns selected number of top classes', required=False, metavar='',action='store')
    parser.add_argument('-i','--image', type = str, default = 'flowers/test/1/image_06743.jpg', help = 'image to predict', required=False, metavar='',action='store')
    return parser.parse_args()