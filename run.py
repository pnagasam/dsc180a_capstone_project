import sys
sys.path.insert(0,'src')

def main(targets):
    if 'data' in targets:
        from data import make_dataset
    if 'features' in targets:
        from features import make_features
    if 'train' in targets:
        from models import train_model
    if 'predict' in targets:
        from models import predict_model
    if 'visualize' in targets:
        from visualization import visualize


if __name__ == "__main__":
    targets = sys.argv[1:]
    main(targets)
