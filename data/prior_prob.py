import config as CFG
import json
import pickle

def generate_prior_proba(root=CFG.pill_root):
    path = root + 'data_train/instances_train.json'
    with open(path, 'r') as f:
        data = json.load(f)
    props = [0 for _ in range(CFG.n_classes)]
    instances = data['annotations']
    for instance in instances:
        props[instance['category_id']] += 1
    
    props = [p / sum(props) for p in props]
    print(props)
    pickle.dump(props, open(root + 'prior_prob.pkl', 'wb'))

if __name__ == '__main__':
    generate_prior_proba()