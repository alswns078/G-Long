import os
import random
import logging
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from config import get_args
from src.dataloader import MSC

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    cudnn.deterministic = False

def get_logger(filename=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    if filename:
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(handler)
    return logger

if __name__ == "__main__":
    args = get_args()
    set_seed(42)
    
    logger = get_logger(f"logs/{args.log_name}")
    logger.info(">>> Initializing G-Long Evaluation on MSC ...")
    
    if args.dataset == "msc":
        runner = MSC(args, logger)
        runner.evaluation()
    else:
        logger.error("Error: This repository currently supports MSC dataset only.")