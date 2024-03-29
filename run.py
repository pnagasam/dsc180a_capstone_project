import os
import shutil
import argparse
import json
import src.data as data
import src.train as train
import src.OT as OT
import src.eval as _eval
import src.viz as viz

DUMMY_SIZE = 1000

def main(args):
    
    if args.exec_type in ['data', 'train', 'OT', 'eval', 'viz', 'all']:
        
        if args.clean:
            print("I'm not deleting the data. It took 2 hours to download! Delete it yourself if you must.")
        else:

            print('Loading Data...')
            
            dataset, metadata = data.load_dataset()
            
            print('Data Loaded.')

    if args.exec_type in ['test']:

        if args.clean:
            print('--clean argument recieved for testing purposes. Exiting...')
            quit()

        else:
            print("Generating Dummy Data...")

            dataset, metadata = data.generate_dummmy_data(DUMMY_SIZE)

            print("Done.")
        
    if args.exec_type in ['train', 'all', 'test']:
            
        train_config = json.load(open('config/train.json'))
        
        if args.clean:

            print("Cleaning Models...")
            try:
                shutil.rmtree(os.path.join(train_config['save_path']))
            except FileNotFoundError:
                print("No Models Found.")
        else:

            fmt = lambda x: 'X' if x else ' '
            print(f"Training Model(s)...     [{train_config['country']}: urban ({fmt(train_config['urban'])}), rural ({fmt(train_config['rural'])})]")
            
            train.go(dataset, metadata, train_config)
            
            print('Finished Training.')
            
    if args.exec_type in ['OT', 'all', 'test']:
        
        OT_config = json.load(open('config/OT.json'))

        if args.clean:

            print("Cleaning OT...")
            try:
                shutil.rmtree(os.path.join(OT_config['save_path']))
            except FileNotFoundError:
                print("No OT Found.")
        else:
        
            print(f"Computing OT...     [{OT_config['source_country']} -> {OT_config['target_country']}]")
            
            OT.go(dataset, metadata, OT_config)
            
            print('Finished OT.')
        
    if args.exec_type in ['eval', 'all', 'test']:

        eval_config = json.load(open('config/eval.json'))

        if args.clean:

            print("Cleaning Results...")
            try:
                shutil.rmtree(os.path.join(eval_config['save_path']))
            except FileNotFoundError:
                print("No Results Found.")
        else:
        
            train_config = json.load(open('config/train.json'))
            
            OT_config = json.load(open('config/OT.json'))
            
            fmt = lambda x: 'X' if x else ' '
            print(f"Evaluating Model(s)...     [{eval_config['source_country']} -> {eval_config['target_country']}: urban ({fmt(eval_config['urban'])}), rural ({fmt(eval_config['rural'])})]")
            
            _eval.go(dataset, metadata, eval_config, train_config, OT_config)
            
            print('Finished Evaluating.')
        
    if args.exec_type in ['viz', 'all', 'test']:
        
        viz_config = json.load(open('config/viz.json'))

        if args.clean:
            print("Cleaning Visualizations...")
            try:
                shutil.rmtree(os.path.join(viz_config['save_path']))
            except FileNotFoundError:
                print("No Visualizations Found.")
        else:
            
            eval_config = json.load(open('config/eval.json'))
            
            train_config = json.load(open('config/train.json'))
            
            OT_config = json.load(open('config/OT.json'))
            
            print(f"Beginning Visualization Script...     [{viz_config['source_country']} -> {viz_config['target_country']}]")
            
            viz.go(dataset, metadata, viz_config, train_config, eval_config, OT_config)

            print("Finished Visualizations.")

        
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("exec_type")
    parser.add_argument("--clean", "-c", action='store_true', help="cleans relevant directories")
    
    args = parser.parse_args()

    main(args)
