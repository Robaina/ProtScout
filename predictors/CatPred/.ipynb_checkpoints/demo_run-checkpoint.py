"""
Enzyme Kinetics Parameter Prediction Script

This script predicts enzyme kinetics parameters (kcat, Km, or Ki) using a pre-trained model.
It processes input data, generates predictions, and saves the results.
"""

import os
import pandas as pd
import numpy as np
from rdkit import Chem
import argparse
import subprocess
import torch
from pathlib import Path
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')
# Suppress specific PyTorch warnings
torch.set_warn_always(False)

def setup_model_weights(weights_dir=None):
    """Set up ESM model weights from local directory if provided."""
    if weights_dir is None:
        return True
        
    weights_path = Path(weights_dir)
    if not weights_path.exists():
        print(f"Warning: Weights directory {weights_dir} does not exist")
        return False
        
    cache_dir = Path('/root/.cache/torch/hub/checkpoints')
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    
    weight_files = [
        'esm2_t33_650M_UR50D.pt',
        'esm2_t33_650M_UR50D-contact-regression.pt'
    ]
    
    success = True
    for filename in weight_files:
        src_file = weights_path / filename
        dst_file = cache_dir / filename

        if not src_file.exists():
            print(f"Warning: ESM weight file {filename} not found in {weights_dir}")
            success = False
            continue

        try:
            import shutil
            shutil.copy2(src_file, dst_file)
            print(f"Copied {filename} to PyTorch cache")
        except Exception as e:
            print(f"Error copying {filename}: {str(e)}")
            success = False

    return success

def create_prediction_files(parameter, input_file_path):
    """Process input data and create necessary files for prediction."""
    df = pd.read_csv(input_file_path)
    smiles_list = df.SMILES
    seq_list = df.sequence
    smiles_list_new = []

    # Process SMILES strings
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            smi = Chem.MolToSmiles(mol)
            if parameter == 'kcat' and '.' in smi:
                smi = '.'.join(sorted(smi.split('.')))
            smiles_list_new.append(smi)
        except:
            print(f'Invalid SMILES input in input row {i}')
            print('Correct your input! Exiting..')
            return None, None, None

    # Validate enzyme sequences
    valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
    for i, seq in enumerate(seq_list):
        if not set(seq).issubset(valid_aas):
            print(f'Invalid Enzyme sequence input in row {i}!')
            print('Correct your input! Exiting..')
            return None, None, None

    # Save processed input
    input_file_new_path = os.path.join('/input', f'processed_{os.path.basename(input_file_path)}')
    df['SMILES'] = smiles_list_new
    df.to_csv(input_file_new_path, index=False)
    
    records_file = input_file_new_path.replace('.csv', '.json')
    output_file = os.path.join('/output', f'predictions_{os.path.basename(input_file_path)}')
    
    return input_file_new_path, records_file, output_file

def run_prediction(parameter, input_file, records_file, output_file, use_gpu):
    """Run the prediction process."""
    checkpoint_dir = f'/app/catpred/production_models/{parameter}/'
    
    # Create PDB records
    subprocess.run([
        'python3', './scripts/create_pdbrecords.py',
        '--data_file', input_file,
        '--out_file', records_file
    ], check=True)
    
    # Set GPU/CPU environment variable
    env = os.environ.copy()
    env['PROTEIN_EMBED_USE_CPU'] = '0' if use_gpu else '1'

    # Run prediction
    subprocess.run([
        'python3', './predict.py',
        '--test_path', input_file,
        '--preds_path', output_file,
        '--checkpoint_dir', checkpoint_dir,
        '--uncertainty_method', 'mve',
        '--smiles_column', 'SMILES',
        '--individual_ensemble_predictions',
        '--protein_records_path', records_file
    ], env=env, check=True)
    
    return output_file

def process_predictions(parameter, output_file):
    """Process and format prediction results."""
    df = pd.read_csv(output_file)
    pred_col, pred_logcol, pred_sd_totcol = [], [], []
    pred_sd_aleacol, pred_sd_epicol = [], []

    unit = 'mM' if parameter != 'kcat' else 's^(-1)'
    target_col = {
        'kcat': 'log10kcat_max',
        'km': 'log10km_mean',
        'ki': 'log10ki_mean'
    }[parameter]

    unc_col = f'{target_col}_mve_uncal_var'
    
    for _, row in df.iterrows():
        model_cols = [col for col in row.index if col.startswith(target_col) and 'model_' in col]
        
        unc = row[unc_col]
        prediction = row[target_col]
        prediction_linear = np.power(10, prediction)
        
        model_outs = np.array([row[col] for col in model_cols])
        epi_unc = np.var(model_outs)
        alea_unc = unc - epi_unc
        
        pred_col.append(prediction_linear)
        pred_logcol.append(prediction)
        pred_sd_totcol.append(np.sqrt(unc))
        pred_sd_aleacol.append(np.sqrt(alea_unc))
        pred_sd_epicol.append(np.sqrt(epi_unc))

    df[f'Prediction_({unit})'] = pred_col
    df['Prediction_log10'] = pred_logcol
    df['SD_total'] = pred_sd_totcol
    df['SD_aleatoric'] = pred_sd_aleacol
    df['SD_epistemic'] = pred_sd_epicol

    # Save final results
    final_output = os.path.join('/output', f'final_{os.path.basename(output_file)}')
    df.to_csv(final_output, index=False)
    print(f'Results saved to: {final_output}')
    
    return df

def main(args):
    """Main function to run the prediction process."""
    print('Starting prediction process...')
    
    # Set up ESM model weights if directory provided
    if args.weights_dir:
        if not setup_model_weights(args.weights_dir):
            print("Warning: Could not set up ESM weights, will download from source")
    
    # Process input and create necessary files
    input_file, records_file, output_file = create_prediction_files(args.parameter, args.input_file)
    if input_file is None:
        return

    print('Running predictions (this will take a while)...\n')
    
    try:
        # Run prediction
        output_file = run_prediction(args.parameter, input_file, records_file, output_file, args.use_gpu)
        
        # Process and save results
        process_predictions(args.parameter, output_file)
        
    except subprocess.CalledProcessError as e:
        print(f'Error during prediction process: {str(e)}')
        return
    except Exception as e:
        print(f'Unexpected error: {str(e)}')
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict enzyme kinetics parameters.")
    parser.add_argument("--parameter", type=str, choices=["kcat", "km", "ki"], required=True,
                      help="Kinetics parameter to predict (kcat, km, or ki)")
    parser.add_argument("--input_file", type=str, required=True,
                      help="Path to the input CSV file")
    parser.add_argument("--use_gpu", action="store_true",
                      help="Use GPU for prediction (default is CPU)")
    parser.add_argument("--weights_dir", type=str,
                      help="Directory containing ESM weight files (optional)")

    args = parser.parse_args()
    args.parameter = args.parameter.lower()

    main(args)