# COS_598L_final_project
Final project for COS 598L: Machine Learning for Structural Biology. Here, I designed a new classifier of protein:Imatinib interactions that can be used to rapidly search the proteome for candidate binding pockets.

First, we downloaded the list of PDB ids with at least one protein and Imatinib (STI molecule id) using the advanced search tool on the PDB. From here, we deduplicated the set of Imatinib binders using the script `define_imatinib_pockets/count_number_of_unique_proteins.py`, which separated the unique structures into `define_imatinib_pockets/unique_imatinib_structures_apr15_2026`.

To calculate the distribution of residue contacts to Imatinib molecules in the PDB, we used `define_pockets.py` which uses the deduplicated structure set `define_imatinib_pockets/unique_imatinib_structures_apr15_2026`. This script produces `define_imatinib_pockets/Figure_1b.png`.

