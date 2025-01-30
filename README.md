## Rank protein sequences based on properties predicted by AI

## Publications
1. https://pubs.acs.org/doi/10.1021/acs.jcim.2c01139

### TODO:

- Optimize containers to enable input of protein embeddings so ESM-2 embeddings computed once
- Not all models use ESM-2, tho, some use ESM-1v. Specifically:
    - Temberture: ProtBERT ('Rostlab/prot_bert_bfd')
    - GATSol: ESM-1b (esm1b_t33_650M_UR50S, esm1b_t33_650M_UR50S-contact-regression)
    - EpHod: ESM-1v (esm1v_t33_650M_UR90S_1, ESM1v-RLATtr)
    - CatPred: ESM-2 (esm2_t33_650M_UR50D, esm2_t33_650M_UR50D-contact-regression)
    - DiffDock: ESM-2 (esm2_t33_650M_UR50D, esm2_t33_650M_UR50D-contact-regression)
    - ThermoMPNN: Custom Graph-based neural network
    - GeoPoc: ESM-2 (esm2_t33_650M_UR50D, esm2_t33_650M_UR50D-contact-regression), user can input embeddings directly

__TARGET__: update all docker images so user  can input embeddings directly for all using ESM.
