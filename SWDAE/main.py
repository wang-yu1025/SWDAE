
import argparse

from models.dawn import *
import models.dawn
from dataloader2 import *
from train import *
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

writer = SummaryWriter('./runs/experiment1')

parser = argparse.ArgumentParser(description='Train Deep SVDD model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num_epochs', '-e', type=int, default=500, help='Num of epochs to Deep SVDD train')
parser.add_argument('--lr', '-lr', type=float, default=0.0001, help='learning rate for model')
parser.add_argument('--weight_decay', '-wd', type=float, default=5e-7, help='weight decay for model')
parser.add_argument('--lr_milestones', '-lr_mile', type=list, default=[50], help='learning rate milestones')
parser.add_argument('--batch_size', '-bs', type=int, default=32, help='batch size')
parser.add_argument('--latent_dim', '-ld', type=int, default=32, help='latent dimension')
parser.add_argument('--normal_class', '-cls', type=int, default=7, help='Set the normal class')
parser.add_argument('--split_data', default=0, type=int,help='take a limited dataset for training')
parser.add_argument('--checkpoint_flag', type=int, default=0, help='Deep SVDD\'s checkpoint flag')
#---------------dwnn模型参数
# parser_dwnn = subparsers.add_parser('dawn')
parser.add_argument("--regu_details", default=0.1, type=float)
parser.add_argument("--regu_approx", default=0.1, type=float)
parser.add_argument("--levels", default=3, type=int)
parser.add_argument("--first_conv", default=128, type=int)
parser.add_argument("--lrdecay", default=[150,225], type=list)
parser.add_argument(
	"--classifier", default='mode1', choices=['mode1', 'mode2','mode3'])
parser.add_argument(
	"--kernel_size", type=int, default=3
)
parser.add_argument(
	"--no_bootleneck", default=False, action='store_true'
)
parser.add_argument(
	"--share_weights", default=False, action='store_true'
)
parser.add_argument(
	"--simple_lifting", default=False, action='store_true'
)
parser.add_argument(
	"--haar_wavelet", default=False, action='store_true'
)
parser.add_argument(
	'--warmup', default=False, action='store_true'
)
args = parser.parse_args()
if __name__ == '__main__':
    
    dataloader_train, dataloader_test = get_mnist(args)

    deep_SVDD = TrainerDeepSVDD(args=args, train_loader=dataloader_train,test_loader = dataloader_test,device=device, R=0.0, nu=0.1, writer=writer)
    c1={}
    test_auroc={}
    if args.checkpoint_flag != 1:
        print("Start Deep SVDD train!")
        Encoder,Decoder,Fc,Fc2,net,c1= deep_SVDD.train()
            # deep_SVDD.get_fea(dataloader_test,net)
        test_auroc=deep_SVDD.test(net,Encoder,Decoder,Fc,Fc2,c1)
    else:
        print("best model checkpoint loaded!!!")
        checkpoint_index = 100
        model_path = './weights/best.pth'
        # deep_SVDD.checkpoint_load(model_path,dataloader_test,dataloader_train)
        net = DAWN(num_classes=2,
                big_input=0,
                first_conv=args.first_conv,
                number_levels=args.levels,
                kernel_size=args.kernel_size,
                no_bootleneck=args.no_bootleneck,
                classifier=args.classifier,
                share_weights=args.share_weights,
                simple_lifting=args.simple_lifting,
    #                 COLOR=USE_COLOR,
                regu_details=args.regu_details,
                regu_approx=args.regu_approx,
                haar_wavelet=args.haar_wavelet
                )
        checkpoint = torch.load(model_path, map_location='cpu')  # 加载模型文件，pt, pth 文件都可以；
            # if torch.cuda.device_count() > 1:
        # 如果有多个GPU，将模型并行化，用DataParallel来操作。这个过程会将key值加一个"module. ***"。
            # net = nn.DataParallel(net) 
        net.load_state_dict(checkpoint) # 接着就可以将模型参数load进模型。
        c1 = torch.Tensor(state_dict['center1']).to(self.device)

        test_auroc1=deep_SVDD.test(net,dataloader_test,dataloader_train)
        print("\nTest AUROC1: {:.2f}\n".format(test_auroc1 * 100))



    
    writer.flush()
    writer.close()