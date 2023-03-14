import os
import argparse
import json
import src.data as data
import src.train as train
import src.OT as OT
import src.eval as _eval
import src.viz as viz

def main(args):
    
    
    if args.exec_type in ['data', 'train', 'OT', 'eval', 'viz', 'all']:
        
        print('Loading Data...')
        
        dataset, metadata = data.load_dataset()
        
        print('Data Loaded.')
    
    if args.exec_type in ['train', 'all']:
        
        train_config = json.load(open('config/train.json'))
        
        fmt = lambda x: 'X' if x else ' '
        print(f"Training Model(s)...     [{train_config['country']}: urban ({fmt(train_config['urban'])}), rural ({fmt(train_config['rural'])})]")
        
        train.go(dataset, metadata, train_config)
        
        print('Finished Training.')
            
    if args.exec_type in ['OT', 'all']:
        
        OT_config = json.load(open('config/OT.json'))
        
        print(f"Computing OT...     [{OT_config['source_country']} -> {OT_config['target_country']}]")
        
        OT.go(dataset, metadata, OT_config)
        
        print('Finished OT.')
        
    if args.exec_type in ['eval', 'all']:
        
        
        
        eval_config = json.load(open('config/eval.json'))
        
        train_config = json.load(open('config/train.json'))
        
        OT_config = json.load(open('config/OT.json'))
        
        fmt = lambda x: 'X' if x else ' '
        print(f"Evaluating Model(s)...     [{eval_config['source_country']} -> {eval_config['target_country']}: urban ({fmt(eval_config['urban'])}), rural ({fmt(eval_config['rural'])})]")
        
        _eval.go(dataset, metadata, eval_config, train_config, OT_config)
        
        print('Finished Evaluating.')
        
    if args.exec_type in ['viz', 'all']:
        
        viz_config = json.load(open('config/viz.json'))
        
        eval_config = json.load(open('config/eval.json'))
        
        train_config = json.load(open('config/train.json'))
        
        OT_config = json.load(open('config/OT.json'))
        
        print(f"Beginning Visualization Script...     [{viz_config['source_country']} -> {viz_config['target_country']}]")
        
        viz.go(dataset, metadata, viz_config, train_config, eval_config, OT_config)
        
        
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("exec_type")
    
    args = parser.parse_args()

    main(args)
