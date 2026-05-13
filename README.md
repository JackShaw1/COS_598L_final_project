# COS_598L_final_project
Final project for COS 598L: Machine Learning for Structural Biology. Here, I designed a new classifier of protein:Imatinib interactions that can be used to rapidly search the proteome for candidate binding pockets.

First, we downloaded the list of PDB ids with at least one protein and Imatinib (STI molecule id) using the advanced search tool on the PDB. From here, we deduplicated the set of Imatinib binders using the script `define_imatinib_pockets/count_number_of_unique_proteins.py`, which separated the unique structures into `define_imatinib_pockets/unique_imatinib_structures_apr15_2026`.

To calculate the distribution of residue contacts to Imatinib molecules in the PDB, we used `define_pockets.py` which uses the deduplicated structure set `define_imatinib_pockets/unique_imatinib_structures_apr15_2026`. This script produces `define_imatinib_pockets/Figure_1b.png`.

Next, we trained a classifier of Imatinib binding pockets using the scripts in `train_classifier`. Specifically, we ran `train_classifier/train.py`, which saves `train_classifier/pocket_classifier.pt`. The final model `train_classifier/pocket_classifier_final.pt` was used to survey the proteome for the results reported in the paper.

Using this final classifier, we scanned the entire AlphaFold Database's human proteome for potential Imatinib binding pockets. To reproduce our analysis, first download the AlphaFold Database's human proteome .tar using `train_classifier/alphafold_human_proteome/download_human_proteome.py`. Then, run `scan_proteome/scan_proteome.py` to start recording proteins and the specific residue indices for candidate Imatinib binding pockets (the script keeps track of all pockets with a score of at least 0.5). Table S1, which contains the final proteome screening results, is equivalent to the file `scan_proteome/imatinib_pocket_hits.csv`.

Lastly, we used Boltz-2 via Modal to produce folds for 72 candidates nominated by our classifier, as well as 37 control samples. These structures were produced using the folding scripts in `boltz-2_comparison`, and the results can be found in our [Dropbox](https://www.dropbox.com/scl/fo/264f5bcgezlwr80igw9dl/ALpYGwCrhNg4cl0OgnAGAVs?rlkey=2t0d263phxaduklxaw8wpzo77&st=wraq7vt7&dl=0). To compare our classifier's scores to the Boltz-2 binding affinity-based interaction probabilities, use the script `boltz-2_comparison/analyze.py`.

The public link for this repo is https://github.com/JackShaw1/COS_598L_final_project
