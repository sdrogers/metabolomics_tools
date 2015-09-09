MultiAlignCmd parameters

HDP
-d /home/joewandy/Dropbox/Project/documents/new_measure_experiment/input_data/P1/100 -gt /home/joewandy/Dropbox/Project/documents/new_measure_experiment/input_data/P1/ground_truth/ground_truth_100.txt -v -method myHdp -experimentType hdp -scoringMethod hdpmassrtjava -hdpLocalRtClusterStdev 2 -hdpGlobalRtClusterStdev 50 -hdpMassTol 500 -groupingNSamples 20 -groupingBurnIn 10

Precursor alignment
-d /home/joewandy/Dropbox/Project/documents/new_measure_experiment/input_data/P1/100 -gt /home/joewandy/Dropbox/Project/documents/new_measure_experiment/input_data/P1/ground_truth/ground_truth_100.txt -v -method myPrecursor -experimentType hdp -db /home/joewandy/git/metabolomics_tools/discretisation/database/std1_mols.csv -trans /home/joewandy/git/metabolomics_tools/discretisation/mulsubs/mulsub2.txt -v -binningMassTol 300 -binningRtTol 60
