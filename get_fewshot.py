import os
import json
import torch
import random
import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default ='banking', help='Path to the training file (e.g., train.tsv)')
parser.add_argument('--data_dir', type=str, default ='./datasets', help='Path to the training file (e.g., train.tsv)')
parser.add_argument("--mode",type=str,default="train",help="which data will be used.")
parser.add_argument("--topn",type=int,default=3,help="top n similar sentecne pair.")
args = parser.parse_args()

# load sentence transformers model
model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
model.to("cuda:0")

df = pd.read_csv(os.path.join(args.data_dir,args.dataset,args.mode+'.tsv'), sep='\t')
grouped_data = df.groupby('label')
# Dictionary containing texts grouped by 'label'
data = grouped_data['text'].apply(list).to_dict()
embeddings_per_category = {category: model.encode(texts) for category, texts in data.items()}
num_clusters = len(data.keys())
random_data = {}

for label, group in tqdm(grouped_data):
    examples = group['text'].tolist()
    random_data[label] = random.sample(examples,min(args.topn,len(examples)))
clustered_data = {}

for label, group in tqdm(grouped_data):
    if len(group['text'])==1:
        clustered_data[label] = group['text'].item()
    else:
        group_vectors = embeddings_per_category[label]
        kmeans = KMeans(n_clusters=1, n_init='auto')
        cluster_labels = kmeans.fit_predict(group_vectors)
        
        # Sample 3 points closest to the center from each cluster
        cluster_centers = kmeans.cluster_centers_
        sampled_points = {}
        for i, center in enumerate(cluster_centers):
            cluster_points = np.array([vector for j, vector in enumerate(group_vectors) if cluster_labels[j] == i])
            distances = np.linalg.norm(cluster_points - center, axis=1)
            
            
            sample_indices = np.argsort(distances)[:min(3, len(distances))]
            sampled_points[i] = [group.iloc[np.where(cluster_labels == i)[0][index]]['text'] for index in sample_indices]
        clustered_data[label] = sampled_points[0]

# Finding most similar sentence pairs between categories
most_similar_pairs = {}
categories = list(data.keys())
top_n_similar_pairs = {}

for i in range(len(categories)):
    for j in range(i+1, len(categories)):
        selected_indices = set()
        category1, category2 = categories[i], categories[j]
        emb_a = embeddings_per_category[category1]
        emb_b = embeddings_per_category[category2]
        top_n = min(len(emb_a),len(emb_b),args.topn)


        similarity_matrix = cosine_similarity(emb_a, emb_b)

        indices = np.ndindex(similarity_matrix.shape)

        sorted_indices = sorted(indices, key=lambda x: similarity_matrix[x], reverse=True)
        # Get top-N similar sentence pairs and their similarity scores
        filtered_top_n = []
        for index in sorted_indices:
            sentence_pair = (data[category1][index[0]], data[category2][index[1]])

            if sentence_pair[0] not in selected_indices and sentence_pair[1] not in selected_indices:
                filtered_top_n.append((sentence_pair, similarity_matrix[index]))
                selected_indices.add(sentence_pair[0])
                selected_indices.add(sentence_pair[1])

            if len(filtered_top_n) == top_n:
                break
        # Record the top-N similar sentence pairs and their similarity scores
        top_n_similar_pairs[(category1, category2)] = filtered_top_n


# Output the results
boundary_sample = {}
for pair, similarities in top_n_similar_pairs.items():
    category1, category2 = pair
    print(f"Top-{top_n} similar pairs between {category1} and {category2}:")
    sim_sentences = []
    for i, (sentences, score) in enumerate(similarities, start=1):
        print(f"{i}. Sentences: {sentences}, Similarity Score: {score}")
        sim_sentences.append(list(sentences))
    boundary_sample[category1+'&'+category2] = sim_sentences
    
# Save results to JSON files
with open(os.path.join(args.data_dir,args.dataset,'random_sample.json'), 'w') as f:
    json.dump(random_data,f,indent=2,ensure_ascii=False)
with open(os.path.join(args.data_dir,args.dataset,'center_sample.json'), 'w') as f:
    json.dump(clustered_data,f,indent=2,ensure_ascii=False)
with open(os.path.join(args.data_dir,args.dataset,'boundary_sample.json'),'w') as f:
    json.dump(boundary_sample,f,indent=2,ensure_ascii=False)

