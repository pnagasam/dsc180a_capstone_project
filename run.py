import sys
import argparse
#from src.models.train_model import train_model
#from src.models.predict_model import predict_model
from src.models.train import predict_model
from os.path import join, abspath

sys.path.insert(0,'src')

def main(args):
    fp_test = join(abspath(join(__file__, "..")),"test/")
    fp_meta = join(fp_test, "dhs_metadata.csv")
    fp_ims = join(fp_test, "images/")
    #clf, train_hists = train_model(args.train_country)
    #predict_model(clf, args.predict_country, train_hists)
    predict_model(fp_ims, fp_meta)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    #parser.add_argument("--train-country", default='angola', type=str, 
    #            help="The country to train the model on.")
    #parser.add_argument("--predict-country", default='benin', type=str,
    #            help="The country to predict the model on.")
    parser.add_argument("--test", action="store_true",
                help="Runs test script on the program.")
    
    args = parser.parse_args()

    main(args)
