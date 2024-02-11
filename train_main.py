from transformers import HfArgumentParser
import ujson

from utils.logger import Logger
from config import MyTrainArugment


def get_argumets(arg_parser) -> MyTrainArugment:

    return arg_parser.parse_args_into_dataclasses()[0]
    

if __name__ == '__main__':

    arg_parser = HfArgumentParser(MyTrainArugment)
    train_args: MyTrainArugment = get_argumets(arg_parser)

    log = Logger('train_agrs', save2file=True)
    log.info(ujson.dumps(train_args.__dict__, ensure_ascii=False, indent=4), std_out=True, save_to_file=True)
    
    # 预训练、sft、dpo入口
    match train_args.train_type:
        case 'pre':
            from train_model.pre_train import pre_train
            pre_train(train_args)

        case 'sft':
            from train_model.sft import sft_train
            sft_train(train_args)

        case 'dpo':
            from train_model.dpo import dpo_train
            dpo_train(train_args)

        case _:
            raise ValueError(f'args `train_type` must be in (`pre`, `sft`, `dpo`), but got `{train_args.train_type}`')
   

    
    