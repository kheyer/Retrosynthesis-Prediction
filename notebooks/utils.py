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


def create_reaction_variants(molecule):
    # creates a variant of the input for each reaction type token
    reactions = ['<RX_1>', '<RX_2>', '<RX_3>', '<RX_4>', '<RX_5>', '<RX_6>',
                 '<RX_7>', '<RX_8>', '<RX_9>', '<RX_10>']
    rxns = [i + ' ' + molecule for i in reactions]
    
    return rxns

def molecules_to_file(molecules, filename):
    # writes molecules to text files
    with open(f'MCTS_data/{filename}.txt', 'w') as out:
        for mol in molecules:
            out.write(mol + '\n')

def smile_valid(smile):
    # determines of a predicted SMILES is valid
    s = ''.join(smile.split(' '))
    smile_check = AllChem.MolFromSmiles(s)
    
    if smile_check:
        return True
    else:
        return False

def canonicalize_smiles(smiles):
    # converts a SMILES string to canonical form
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return ''

def process_smile(smile):
    # Processes tokenized SMILES string into a form that RDKit accepts
    if '> ' in smile:
        # SMILES may have reaction token, this strips the reaction token
        smile = smile.split('> ')[1]
        
    smile = ''.join(smile.split(' '))
    return smile

def smile_to_mol(smile):
    # converts text SMILES into RDKit Mol object
    return Chem.MolFromSmiles(smile)

def canonicalize_prediction(smiles):
    # converts tokenized predicted SMILES into canonicalized form
    smiles = canonicalize_smiles(process_smile(smiles))
    
    if smiles == '':
        return smiles
    else:
        return ' '.join([i for i in smiles])

def create_prediction_df(molecule_variants, predictions, scores):
    # Create dataframe of predictions
    # molecule variants - variants of the source product with different reaction tokens
    # predictions - list of lists of predicted SMILES for each molecule variant
    # scores - list of lists for scores corresponding to predictions
    dfs = []

    # for ech variant, create dataframe of input, products, mechanisms and predictions
    for i in range(len(molecule_variants)):
        df_iter = pd.DataFrame({f'Prediction' : predictions[i], f'Score' : scores[i]})
        df_iter[f'Score'] = df_iter[f'Score'].map(lambda x: x.item())
        df_iter['Input'] = molecule_variants[i]
        df_iter['Product_Molecule'] = df_iter.Input.map(lambda x: x.split('> ')[1])
        df_iter['Mechanism'] = df_iter.Input.map(lambda x: x.split(' ')[0])
        dfs.append(df_iter)

    df = pd.concat(dfs, axis=0)
    df = df.reset_index(drop=True)
    
    return df

def clean_predictions(df):
    # cleans a dataframe of predictions

    # drops all invalid SMILES predictions
    df = df[df[f'Prediction'].map(lambda x: smile_valid(x))]
    df = df.reset_index(drop=True)

    # canonicalizes predictions
    df[f'Prediction'] = df[f'Prediction'].map(lambda x: canonicalize_prediction(x))

    # applys stoichiometry check
    df = df[df.apply(lambda row: check_stoichiometry(row['Prediction'], row['Product_Molecule']), axis=1)]
    
    # removes trivial predictions where product molecule is contained in predicted reactants
    df = df[~df.apply(lambda row: row[f'Product_Molecule'] in row[f'Prediction'], axis=1)]
    df = df.reset_index(drop=True)
    
    return df

def process_predictions(df):
    # Clean and score prediction dataframe
    df = clean_predictions(df)
    df.reset_index(inplace=True, drop=True)
    df = score_predictions(df)
    
    return df

def score_predictions(df):
    # apply scoring function to predictions
    df[f'Prediction_Score'] = df.apply(lambda row: 
                                     heuristic_scoring(row[f'Product_Molecule'], 
                                                       row[f'Prediction'], 
                                                       row[f'Score']), axis=1)
    return df