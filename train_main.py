from transformers import HfArgumentParser, TrainingArguments
import ujson

from loguru import logger
from config import CustomArugments


def get_argumets() -> tuple[CustomArugments, TrainingArguments]:
    arg_parser = HfArgumentParser(dataclass_types=(CustomArugments, TrainingArguments))
    return arg_parser.parse_args_into_dataclasses()
    

if __name__ == '__main__':

    cust_args: CustomArugments
    train_args: TrainingArguments
    cust_args, train_args = get_argumets()

    logger.info(cust_args.__dict__)
 
    # 预训练、sft、dpo入口
    match cust_args.train_type:
        case 'pre':
            from train_model.pre_train import pre_train
            pre_train(cust_args=cust_args, train_args=train_args)

        case 'sft':
            from train_model.sft import sft_train
            sft_train(cust_args=cust_args, train_args=train_args)

        case 'dpo':
            from train_model.dpo import dpo_train
            dpo_train(cust_args=cust_args, train_args=train_args)

        case _:
            raise ValueError(f'args `train_type` must be in (`pre`, `sft`, `dpo`), but got `{train_args.train_type}`')
   

    
    