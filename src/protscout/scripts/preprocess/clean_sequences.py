import os
import logging
import argparse
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Union, Optional
import pyfastx

def setup_logger(log_file=None):
    """
    Set up logging configuration
    
    Args:
        log_file (str, optional): Path to the log file. If None, no logging is performed.
    """
    if log_file:
        # Create directory for log file if it doesn't exist
        log_dir = Path(log_file).parent
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure logging to write to file
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.FileHandler(log_file)]
        )
    
    return logging.getLogger(__name__)

def clean_fasta(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    remove_asterisk: bool = True,
) -> None:
    """
    Clean FASTA headers and optionally remove asterisk symbols.
    
    Args:
        input_path: Path to input FASTA file
        output_path: Path to output FASTA file. If None, overwrites input file
        remove_asterisk: Whether to remove asterisk symbols from sequences
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        # Use pyfastx to read sequences - works with both .gz and uncompressed files
        fa = pyfastx.Fastx(str(input_path))
        
        with open(output_path, 'w') as out:
            for name, seq in fa:
                # Simplify header: keep only first part before whitespace
                clean_name = name.split()[0]
                clean_name = clean_name.replace(" ", "_")  # Replace spaces with underscores
                clean_name = clean_name.replace("|", "___")  # Replace pipe characters with triple underscores
                
                # Remove asterisk if requested
                if remove_asterisk:
                    seq = seq.replace("*", "")
                    
                # Write in FASTA format
                out.write(f">{clean_name}\n{seq}\n")

    except Exception as e:
        raise ValueError(f"Error processing FASTA file: {e}")

def remove_duplicate_sequences(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    use_header: bool = False,
    remove_asterisk: bool = True,
) -> tuple[int, int]:
    """
    Remove duplicate sequences from a FASTA file.
    
    Args:
        input_path: Path to input FASTA file
        output_path: Path to output FASTA file. If None, overwrites input file
        use_header: If True, sequences with identical headers are considered duplicates
                   If False, sequences with identical content are considered duplicates
        remove_asterisk: Whether to remove asterisk symbols from sequences
        
    Returns:
        tuple: (total_sequences, unique_sequences)
    """
    input_path = Path(input_path)
    # Initialize temp_output to None to fix the error
    temp_output = None
    
    if output_path is None:
        # Create a temporary file if we're overwriting the input
        temp_output = input_path.with_suffix('.temp.fasta')
        output_path = temp_output
    else:
        output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Dictionary to store seen sequences or headers
    seen = set()
    total_count = 0
    unique_count = 0
    
    try:
        # Use pyfastx to read sequences - efficient for large files
        fa = pyfastx.Fastx(str(input_path))
        
        with open(output_path, 'w') as out:
            for name, seq in fa:
                total_count += 1
                
                # Clean header: keep only first part before whitespace
                clean_name = name.split()[0]
                clean_name = clean_name.replace(" ", "_")  # Replace spaces with underscores
                clean_name = clean_name.replace("|", "___")  # Replace pipe characters with triple underscores
                
                # Remove asterisk if requested
                if remove_asterisk:
                    seq = seq.replace("*", "")
                
                # Determine the key for duplicate checking
                if use_header:
                    # Use header as the key
                    key = clean_name
                else:
                    # Use sequence content as the key (make a hash for efficiency with large sequences)
                    key = hashlib.md5(seq.encode()).hexdigest()
                
                # Only write if we haven't seen this sequence/header before
                if key not in seen:
                    seen.add(key)
                    unique_count += 1
                    out.write(f">{clean_name}\n{seq}\n")
        
        # If we're overwriting the input file, replace it with our temp file
        # Fixed: only check temp_output if it's not None
        if temp_output is not None:
            temp_output.replace(input_path)
            
    except Exception as e:
        # Fixed: only check temp_output if it's not None
        if temp_output is not None and temp_output.exists():
            temp_output.unlink()  # Clean up temp file on error
        raise ValueError(f"Error processing FASTA file: {e}")
    
    return total_count, unique_count

def process_fasta_files(input_dir, output_dir, logger, prefix="cleaned_", remove_duplicates=False, use_header=False, remove_asterisk=True):
    """
    Process FASTA files from input directory and save to output directory
    
    Args:
        input_dir: Directory containing input FASTA files
        output_dir: Directory to save cleaned FASTA files
        logger: Logger instance
        prefix: Prefix to add to output filenames (default: "cleaned_")
        remove_duplicates: Whether to remove duplicate sequences
        use_header: If True and remove_duplicates is True, sequences with identical headers are considered duplicates
                   If False and remove_duplicates is True, sequences with identical content are considered duplicates
        remove_asterisk: Whether to remove asterisk symbols from sequences
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Include both compressed and uncompressed files - pyfastx handles both
    fasta_files = [f for f in os.listdir(input_dir) if f.endswith(('.fasta', '.fa', '.fna', '.faa', '.gz'))]
    logger.info(f"Starting processing of {len(fasta_files)} files")
    
    for fasta_file in fasta_files:
        try:
            input_path = os.path.join(input_dir, fasta_file)
            # Remove .gz extension for output filename if present, but keep the fasta extension
            base_name = fasta_file[:-3] if fasta_file.endswith('.gz') else fasta_file
            output_path = os.path.join(output_dir, f"{prefix}{base_name}")
            
            logger.info(f"Processing file: {fasta_file}")
            
            if remove_duplicates:
                # Remove duplicate sequences
                logger.info(f"Removing duplicate sequences from {fasta_file}")
                total, unique = remove_duplicate_sequences(
                    input_path=input_path,
                    output_path=output_path,
                    use_header=use_header,
                    remove_asterisk=remove_asterisk
                )
                logger.info(f"Found {total} sequences, kept {unique} unique sequences ({total-unique} duplicates removed)")
            else:
                # Just clean the FASTA file
                clean_fasta(
                    input_path=input_path,
                    output_path=output_path,
                    remove_asterisk=remove_asterisk
                )
            
            logger.info(f"Successfully processed {fasta_file} -> {output_path}")
            
        except Exception as e:
            logger.error(f"Error processing {fasta_file}: {str(e)}", exc_info=True)
    
    logger.info("Processing completed")

def parse_arguments():
    """Set up and parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Clean FASTA files by simplifying headers and optionally removing asterisks and duplicates",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-i", "--input_dir",
        required=True,
        help="Directory containing input FASTA files"
    )
    
    parser.add_argument(
        "-o", "--output_dir",
        required=True,
        help="Directory to save cleaned FASTA files"
    )
    
    parser.add_argument(
        "-l", "--log_file",
        help="Path to log file (optional). If not provided, no logging will be performed",
        default=None
    )
    
    parser.add_argument(
        "--keep_asterisk",
        action="store_true",
        help="Keep asterisk symbols in sequences (default: remove asterisks)"
    )
    
    parser.add_argument(
        "--prefix",
        type=str,
        default="cleaned_",
        help="Prefix to add to output filenames (default: 'cleaned_')"
    )
    
    parser.add_argument(
        "--remove_duplicates",
        action="store_true",
        help="Remove duplicate sequences from FASTA files"
    )
    
    parser.add_argument(
        "--use_header_for_dedup",
        action="store_true",
        help="When removing duplicates, consider sequences with identical headers (rather than sequence content) as duplicates"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    # Setup logger
    logger = setup_logger(args.log_file)
    
    try:
        # Process the files
        process_fasta_files(
            args.input_dir, 
            args.output_dir, 
            logger,
            prefix=args.prefix,
            remove_duplicates=args.remove_duplicates,
            use_header=args.use_header_for_dedup,
            remove_asterisk=not args.keep_asterisk
        )
    except Exception as e:
        if logger:
            logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise