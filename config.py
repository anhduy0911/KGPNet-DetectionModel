# path
detection_folder = '/home/huyen/projects/duyna/data/2904_VAIPE-Matching/pills/'
detection_root = 'data/'
prescription_folder = 'data/pills/data_train_ai4vn/prescription/label'
base_log = 'logs/'
log_dir_data = 'logs/data/'
graph_ebds_path = 'data/graph/graph_ebd.pt'
graph_root = 'data/graph/'
pill_root = 'data/pills/'
warmstart_path = 'logs/baseline/'
g_warmstart_path = 'logs/graph/'
# statistics
n_classes = 108
n_workers = 8
seed = 9110
max_iters = 60000
test_period = 5000
cpt_frequency = 5000
linking_loss_weight = 0.1
num_head_gtn = 10
topk_neighbor = 10