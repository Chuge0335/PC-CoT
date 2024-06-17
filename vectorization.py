from sentence_transformers import SentenceTransformer
import pickle
import pandas as pd
import argparse
import csv
import os

def main():

    # read data
    train_dataset = pd.read_csv(os.path.join(args.data_dir,args.dataset,'test.tsv'), sep='\t', quoting=csv.QUOTE_NONE)
    save_path = os.path.join(args.data_dir,args.dataset,'label_vectors.pkl')
    # get the label names
    labels = list(set(train_dataset['label'].tolist()))
    model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
    label_vectors = model.encode(labels)
    # label-to-vector mappings
    label_to_vector = {label: vector.tolist() for label, vector in zip(labels, label_vectors)}

    # save to pickle
    with open(save_path, 'wb') as file:
        pickle.dump(label_to_vector, file)

    print(len(labels))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get the labels vector representation.')
    parser.add_argument('--dataset', type=str, default ='banking', help='Path to the training file (e.g., train.tsv)')
    parser.add_argument('--data_dir', type=str, default ='./datasets', help='Path to the training file (e.g., train.tsv)')

    args = parser.parse_args()
    main()