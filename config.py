def initialize(args):

    import warnings
    warnings.filterwarnings("ignore") 

    import os
    if args.device=='cpu':
        os.environ['CUDA_VISIBLE_DEVICES']=''

    import torch

    torch.autograd.set_detect_anomaly(False)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    if not torch.cuda.is_available():
        args.device='cpu'
        args.devices=['cpu']
        print('Switch to cpu')

    import numpy as np
    if args.seed!=None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.device!='cpu':
        if args.seed!=None:
            torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(args.device)

    import pykeops

    try:
        pykeops.set_bin_folder(f'.cache/pykeops{pykeops.__version__}/{os.uname().nodename}/')
    except AttributeError:
        pykeops.set_build_folder(f'.cache/pykeops{pykeops.__version__}/{os.uname().nodename}/')
    return args

tmp_dir='/tmp'