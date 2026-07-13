import torch.nn as nn
from Datasets import load_dataset, compute_second_order_neighbors
from tqdm import tqdm
import warnings
import torch
import torch_geometric
import argparse
import sys
from CREST import DimensionNN_V2, GraphTransformer_encoder, MLP_encoder, CREST
from CREST import DimensionNN_V2, CREST
from Utils import dimensional_sample_random, DAD_edge_index, freeze_test, get_embedding

torch.cuda.empty_cache()


def run(args):
    with open(args.log_dir, 'a') as f:
        f.write('\n\n\n')
        f.write(str(args))
    free_gpu_id = args.GPU_ID
    torch.cuda.set_device(int(free_gpu_id))
    dataset = args.dataset
    data_dir = args.datadir
    nb_epochs = args.nb_epochs
    lr = args.lr
    wd = args.wd
    hid_units = args.hid_units
    num_hop = args.num_hop
    activator = nn.PReLU if args.activator == 'PReLU' else nn.ReLU
    torch_geometric.seed.seed_everything(args.seed)
    seed = args.seed
    sample_size = args.sample_size
    feature_signal_dim = args.feature_signal_dim
    losslam_ssl = args.losslam_ssl
    losslam_sig_cross = args.losslam_sig_cross
    if_rand = True if args.if_rand == 'True' else False

    data = load_dataset(dataset, data_dir)[0]
    edge_index = data.edge_index.cuda()
    second_order_edge_index = compute_second_order_neighbors(edge_index)
    second_order_edge_index = second_order_edge_index.cuda()
    data.second_order_edge_index = second_order_edge_index

    dnn = DimensionNN_V2(sample_size, feature_signal_dim * 2, feature_signal_dim, activator)
    mlp = MLP_encoder(feature_signal_dim, hid_units, activator)
    trans = GraphTransformer_encoder(feature_signal_dim, hid_units, activator)
    model = CREST(D_NN=dnn, MLP=mlp, Trans=trans, S_mtd=dimensional_sample_random, sample_size=sample_size)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    if torch.cuda.is_available():
        data = data.cuda()
        model = model.cuda()
    model.update_sample(data.x, data.second_order_edge_index, if_rand=if_rand)
    with tqdm(total=nb_epochs, desc='(T)') as pbar:
        for epoch in range(nb_epochs):
            model.train()
            optimiser.zero_grad()
            z_g, z_t = model(data.x,
                             data.second_order_edge_index)
            loss_ssl = (model.infonce_loss(z_g, z_t) + model.infonce_loss(z_t, z_g)) / 2
            loss_sig_cross = model.dim_loss_fn()
            loss = losslam_ssl * loss_ssl + losslam_sig_cross * loss_sig_cross
            loss.backward()
            optimiser.step()

            pbar.set_postfix({'loss_ssl': loss_ssl.item(),
                              'loss_sig_cross': loss_sig_cross.item()
                              }
                             )
            pbar.update()

    model.eval()
    z = get_embedding(data.x, data.second_order_edge_index, model, num_hop, if_rand=if_rand)

    m, r = freeze_test(z, data.y, train_ratio=0.6, test_ratio=0.2, test_num=20)
    with open(args.log_dir, 'a') as f:
        f.write('\n')
        f.write(' mean: ' + str(m) + ' std: ' + str(r))


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # setting arguments
    parser = argparse.ArgumentParser('CREST')
    parser.add_argument('--dataset', type=str, default='Wisconsin',
                        help="""Dataset name: Cora, CiteSeer, PubMed, dblp, Photo, Computers, CS, Physics,
    ogbn-products, ogbn-arxiv, Wiki, ppi, Cornell, Texas, Wisconsin,
    chameleon, crocodile, squirrel, actor, roman_empire, amazon_ratings,
    minesweeper, tolokers, questions, chameleon_filtered, squirrel_filtered""")
    parser.add_argument('--datadir', type=str, default='../../../datasets/', help='./data/dir/')
    parser.add_argument('--log_dir', type=str, default='./log/logCora.txt', help='./log/dir/')
    parser.add_argument('--GPU_ID', type=int, default=0, help='The GPU ID')
    parser.add_argument('--seed', type=int, default=777, help='seed')

    parser.add_argument('--nb_epochs', type=int, default=1000, help='Number of epochs')  # training epochs
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.00001, help='weight decay')
    parser.add_argument('--activator', type=str, default='PReLU', help='Activator name: PReLU, ReLU')
    parser.add_argument('--if_rand', type=str, default='False', help='feature sample if_rand: True, False')

    parser.add_argument('--hid_units', type=int, default=1024, help='representation size')
    parser.add_argument('--sample_size', type=int, default=183, help='node sample batch size')
    parser.add_argument('--feature_signal_dim', type=int, default=1024, help='feature signal dim')
    parser.add_argument('--losslam_ssl', type=float, default=0.1, help='hyper-parameter of ssl loss')
    parser.add_argument('--losslam_sig_cross', type=float, default=100, help='hyper-parameter of sig_cross loss')

    parser.add_argument('--num_hop', type=int, default=0, help='graph view hop num')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    run(args)
