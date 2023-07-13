import os
import pprint
import neptune
import warnings

from train import Trainer
from utils import fix_seed
from config import getConfig

warnings.filterwarnings('ignore')
args = getConfig()

def main(args):

    print('<---- Training Params ---->')
    pprint.pprint(args)

    # Random Seed
    fix_seed(args.seed)

    if args.logging:
        api = "api_token"
        run = neptune.init_run(project="ID/Project", api_token=api, name=args.experiment, tags=args.tag.split(','))
        run.assign({'parameters':vars(args)})
        exp_num = run._sys_id.split('-')[-1].zfill(3)
    else:
        run = None
        exp_num = args.exp_num

    save_path = os.path.join(args.model_path, exp_num)

    # Create model directory
    os.makedirs(save_path, exist_ok=True)
    Trainer(args, save_path, run).outer_loop()

if __name__ == '__main__':
    main(args)