import argparse

parser = argparse.ArgumentParser(description='hierarchical network')

#======================Code Configs=========================
parser.add_argument('--train_video_list', default='list/ucf_trans_split1_train.txt', type=str) # ucf_trans_split1_train.txt, kinetics_train
parser.add_argument('--test_video_list', default='list/ucf_trans_split1_test.txt', type=str)
parser.add_argument('--root', default='../Datasets/', type=str)
parser.add_argument('--dataset', default='ucf', type=str)
parser.add_argument('--log_dir', default='log', type=str)
parser.add_argument('--model_dir', default='model', type=str)
parser.add_argument('--score_dir', default='scores', type=str)
parser.add_argument('--get_scores', default=False, type=bool)
parser.add_argument('--description', default='collaborate', type=str)
parser.add_argument('--cross', default=False, type=bool)
parser.add_argument('--freeze', default=True, type=bool)



#===================Learning Configs====================
parser.add_argument('--batch_size', default=20, type=int)
parser.add_argument('--workers', default=6, type=int)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=1e-5, type=str)
parser.add_argument('--epoch', default=35, type=int)
parser.add_argument('--lr_step', default=[30, 40], type=int)
parser.add_argument('--print_freq', default=20, type=int)
parser.add_argument('--eval_freq', default=1, type=int)



#====================Model Configs======================
parser.add_argument('--segments', default=1, type=int)# biedong
parser.add_argument('--frames', default=32, type=int)# zhidong zhege
parser.add_argument('--model_depth', default=34, type=int)
parser.add_argument('--stride', default=1, type=int)
parser.add_argument('--accumulation_steps', default=1, type=int)
