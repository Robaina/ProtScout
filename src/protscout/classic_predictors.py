from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis


def compute_basic_protein_properties(
    fasta_path: str, output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Analyze properties of proteins from a FASTA file and save results to TSV.

    Args:
        fasta_path: Path to input FASTA file
        output_path: Optional path to save TSV output (defaults to input path with .tsv extension)

    Returns:
        DataFrame containing protein properties

    Properties calculated:
        - molecular_weight_kda: Mass of protein in kilodaltons, calculated from amino acid composition
        - isoelectric_point: pH at which protein has neutral net charge (affects solubility)
        - extinction_coefficient:
            - reduced: Protein absorption at 280nm without disulfide bonds
            - oxidized: Protein absorption at 280nm with disulfide bonds
            Used for protein concentration measurements in spectrophotometry
        - instability_index:
            Value predicting protein stability in vitro
            < 40: Probably stable
            > 40: Probably unstable
        - flexibility: Average protein chain flexibility based on amino acid composition
            Higher values indicate more flexible regions
            Used in structural predictions

    """
    results: List[Dict] = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        sequence = str(record.seq).replace("*", "")

        try:
            analysis = ProteinAnalysis(sequence)

            # Calculate properties
            properties = {
                "protein_id": record.id,
                "sequence_length": len(sequence),
                "molecular_weight_kda": analysis.molecular_weight() / 1000,
                "isoelectric_point": analysis.isoelectric_point(),
                "extinction_coefficient_reduced": analysis.molar_extinction_coefficient()[
                    0
                ],
                "extinction_coefficient_oxidized": analysis.molar_extinction_coefficient()[
                    1
                ],
                "instability_index": analysis.instability_index(),
                "avg_flexibility": sum(analysis.flexibility())
                / len(analysis.flexibility()),
            }
            results.append(properties)

        except Exception as e:
            print(f"Error processing {record.id}: {str(e)}")
            continue

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save to TSV if path provided
    if output_path is None:
        output_path = str(Path(fasta_path).with_suffix(".tsv"))
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)

    return df
