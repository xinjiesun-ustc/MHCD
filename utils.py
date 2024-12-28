import argparse
from build_graph import build_graph

# class CommonArgParser(argparse.ArgumentParser):
#     def __init__(self):
#         super(CommonArgParser, self).__init__()
#         self.add_argument('--exer_n', type=int, default=835,
#                           help='The number for exercise.')
#         self.add_argument('--knowledge_n', type=int, default=835,
#                           help='The number for knowledge concept.')
#         self.add_argument('--student_n', type=int, default=10000,
#                           help='The number for student.')
#         self.add_argument('--gpu', type=int, default=0,
#                           help='The id of gpu, e.g. 0.')
#         self.add_argument('--epoch_n', type=int, default=20,
#                           help='The epoch number of training')
#         self.add_argument('--lr', type=float, default=0.002,
#                           help='Learning rate')
#         self.add_argument('--test', action='store_true',
#                           help='Evaluate the model on the testing set in the training process.')
#         self.add_argument('--dataset', type=str, default="junyi",
#                           help='the dataset name')

def construct_local_map(dataset):
    local_map = {
        # 'directed_g': build_graph('direct', knowledge_n,dataset=dataset),
        # 'undirected_g': build_graph('undirect', knowledge_n,dataset=dataset),
        'k_from_e': build_graph('k_from_e', 6+6,dataset=dataset),
        'e_from_k': build_graph('e_from_k',6+6,dataset=dataset),
        'u_from_e': build_graph('u_from_e', 6+6,dataset=dataset),
        'e_from_u': build_graph('e_from_u', 6+6,dataset=dataset),
    }
    print(local_map)
    return local_map

