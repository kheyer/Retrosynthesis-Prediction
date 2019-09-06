from __future__ import unicode_literals
from itertools import repeat

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

def translate_file(input_filename, output_filename):
    parser = ArgumentParser(description='translation')
    opts.config_opts(parser)
    opts.translate_opts(parser)
    
    args = f'''-model Experiments/Checkpoints/retrosynthesis_augmented_medium/retrosynthesis_aug_medium_model_step_100000.pt 
                -src MCTS_data/{input_filename}.txt 
                -output MCTS_data/{output_filename}.txt 
                -batch_size 128 
                -replace_unk
                -max_length 200 
                -verbose 
                -beam_size 10 
                -n_best 10 
                -min_length 5 
                -gpu 0'''
    
    opt = parser.parse_args(args)
    translator = build_translator(opt, report_score=True)
    
    src_shards = split_corpus(opt.src, opt.shard_size)
    tgt_shards = repeat(None)
    shard_pairs = zip(src_shards, tgt_shards)
    
    for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
        scores, predictions = translator.translate(
            src=src_shard,
            tgt=tgt_shard,
            src_dir=opt.src_dir,
            batch_size=opt.batch_size,
            attn_debug=opt.attn_debug
            )
        
    return scores, predictions