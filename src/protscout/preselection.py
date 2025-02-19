from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Union, Optional, Set, Tuple
from collections import defaultdict
import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq


class SequenceFilter:
    """
    filter for protein sequences based on multiple criteria including
    completeness, length, and composition.
    """

    def __init__(
        self,
        n_complete: bool = True,
        c_complete: bool = True,
        min_length_percentile: Optional[int] = None,
        max_length_percentile: Optional[int] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ):
        """
        Initialize the sequence filter with configurable parameters.

        Args:
            n_complete: If True, only keep sequences with complete N-terminus
            c_complete: If True, only keep sequences with complete C-terminus
            min_length_percentile: Percentile for minimum length cutoff, ignored if None or if min_length is set
            max_length_percentile: Percentile for maximum length cutoff, ignored if None or if max_length is set
            min_length: Minimum sequence length required. If None, uses percentile if set
            max_length: Maximum sequence length allowed. If None, uses percentile if set
        """
        self.n_complete = n_complete
        self.c_complete = c_complete
        self.min_length_percentile = min_length_percentile
        self.max_length_percentile = max_length_percentile
        self.min_length = min_length
        self.max_length = max_length

        # Common N-terminal starting residues and C-terminal motifs
        self.valid_starts: Set[str] = {"M"}
        self.valid_ends: Set[str] = {"*", "TERM"}

        # Valid amino acids
        self.valid_aa: Set[str] = set("ACDEFGHIKLMNPQRSTVWY*")

    def check_termini(self, header: str, sequence: str) -> bool:
        """
        Check if protein has proper N and C termini using both prodigal partial codes
        and direct sequence inspection.

        Args:
            header: Sequence header that might contain prodigal partial codes
            sequence: Amino acid sequence

        Returns:
            bool: True if termini appear valid
        """
        # First try to use prodigal partial codes
        partial_match = re.search(r"partial=(\d{2})", header)
        if partial_match:
            partial_code = partial_match.group(1)
            n_term_complete = partial_code[0] == "0"
            c_term_complete = partial_code[1] == "0"

            if self.n_complete and not n_term_complete:
                return False
            if self.c_complete and not c_term_complete:
                return False
            return True

        # If no prodigal codes found, check sequence directly
        if self.n_complete and sequence[0] not in self.valid_starts:
            return False
        if self.c_complete and (
            sequence[-1] not in self.valid_ends and sequence[-4:] not in self.valid_ends
        ):
            return False
        return True

    def check_unusual_composition(self, sequence: str) -> bool:
        """
        Check for unusual amino acid composition that might indicate problems with the sequence.

        Args:
            sequence: Amino acid sequence

        Returns:
            bool: True if composition appears normal
        """
        # Count amino acid frequencies
        aa_counts = defaultdict(int)
        for aa in sequence:
            aa_counts[aa] += 1

        # Check for unusually high frequency of any amino acid (>40%)
        sequence_length = len(sequence)
        for count in aa_counts.values():
            if count / sequence_length > 0.4:
                return False

        # Check for unusual characters
        if not set(sequence).issubset(self.valid_aa):
            return False

        return True

    def get_length_thresholds(
        self, sequences: List[str]
    ) -> Optional[Tuple[float, float]]:
        """
        Calculate length thresholds based on percentiles of sequence lengths.
        Returns None if no length filtering is required.

        Args:
            sequences: List of protein sequences

        Returns:
            Optional tuple: (min_length, max_length) or None if no length filtering
        """
        # If all length parameters are None, return None to skip length filtering
        if all(
            param is None
            for param in [
                self.min_length,
                self.max_length,
                self.min_length_percentile,
                self.max_length_percentile,
            ]
        ):
            return None

        # Calculate length thresholds
        lengths = np.array([len(seq) for seq in sequences])

        min_len = (
            self.min_length
            if self.min_length is not None
            else (
                np.percentile(lengths, self.min_length_percentile)
                if self.min_length_percentile is not None
                else float("-inf")
            )
        )
        max_len = (
            self.max_length
            if self.max_length is not None
            else (
                np.percentile(lengths, self.max_length_percentile)
                if self.max_length_percentile is not None
                else float("inf")
            )
        )

        return float(min_len), float(max_len)

    def filter_sequences(self, sequences: Dict[str, str]) -> Dict[str, str]:
        """
        Filter protein sequences based on completeness, length and composition criteria.

        Args:
            sequences: Dictionary with protein headers as keys and sequences as values

        Returns:
            Dictionary containing only the sequences matching all requirements
        """
        # Get length thresholds if any length filtering is required
        length_thresholds = self.get_length_thresholds(list(sequences.values()))

        filtered_sequences = {}

        for header, sequence in sequences.items():
            # Remove stop codon for length calculation
            clean_seq = sequence.rstrip("*")
            seq_length = len(clean_seq)

            # Check basic requirements (termini and composition)
            meets_requirements = self.check_termini(
                header, sequence
            ) and self.check_unusual_composition(sequence)

            # Apply length filtering only if thresholds are set
            if length_thresholds is not None:
                min_len, max_len = length_thresholds
                meets_requirements = meets_requirements and (
                    min_len <= seq_length <= max_len
                )

            if meets_requirements:
                filtered_sequences[header] = sequence

        return filtered_sequences

    def filter_fasta(
        self, input_path: Union[str, Path], output_path: Union[str, Path]
    ) -> Dict[str, Union[int, float, str]]:
        """
        Filter sequences from a FASTA file based on all criteria.

        Args:
            input_path: Path to input FASTA file
            output_path: Path where filtered FASTA file will be written

        Returns:
            Dictionary containing filtering statistics
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Read sequences
        sequences = {
            f">{record.description}": str(record.seq)
            for record in SeqIO.parse(input_path, "fasta")
        }

        # Filter sequences
        filtered_sequences = self.filter_sequences(sequences)

        # Write filtered sequences
        with open(output_path, "w") as f:
            for header, sequence in filtered_sequences.items():
                clean_header = header[1:] if header.startswith(">") else header
                record = SeqRecord(
                    Seq(sequence), id=clean_header.split()[0], description=clean_header
                )
                SeqIO.write(record, f, "fasta")

        return self.get_filtering_stats(len(sequences), len(filtered_sequences))

    def get_filtering_stats(
        self, input_count: int, output_count: int
    ) -> Dict[str, Union[int, float, str]]:
        """
        Get statistics about the filtering process.

        Args:
            input_count: Number of input sequences
            output_count: Number of sequences after filtering

        Returns:
            Dictionary containing filtering statistics
        """
        return {
            "total_sequences": input_count,
            "filtered_sequences": output_count,
            "sequences_removed": input_count - output_count,
            "percentage_kept": round((output_count / input_count) * 100, 1),
            "n_terminal_required": self.n_complete,
            "c_terminal_required": self.c_complete,
            "min_length": (
                str(self.min_length) if self.min_length is not None else "None"
            ),
            "max_length": (
                str(self.max_length) if self.max_length is not None else "None"
            ),
            "min_length_percentile": (
                str(self.min_length_percentile)
                if self.min_length_percentile is not None
                else "None"
            ),
            "max_length_percentile": (
                str(self.max_length_percentile)
                if self.max_length_percentile is not None
                else "None"
            ),
        }


def filter_sequences(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    n_complete: bool = True,
    c_complete: bool = True,
    min_length_percentile: Optional[int] = None,
    max_length_percentile: Optional[int] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
) -> Dict[str, Union[int, float, str]]:
    """
    Convenience function to filter protein sequences from a FASTA file.

    Args:
        input_path: Path to input FASTA file
        output_path: Path where filtered FASTA file will be written
        n_complete: If True, only keep sequences with complete N-terminus
        c_complete: If True, only keep sequences with complete C-terminus
        min_length_percentile: Percentile for minimum length cutoff, ignored if None
        max_length_percentile: Percentile for maximum length cutoff, ignored if None
        min_length: Minimum sequence length required. If set, overrides min_length_percentile
        max_length: Maximum sequence length allowed. If set, overrides max_length_percentile

    Returns:
        Dictionary containing filtering statistics
    """
    sequence_filter = SequenceFilter(
        n_complete=n_complete,
        c_complete=c_complete,
        min_length_percentile=min_length_percentile,
        max_length_percentile=max_length_percentile,
        min_length=min_length,
        max_length=max_length,
    )

    stats = sequence_filter.filter_fasta(input_path, output_path)
    return stats
