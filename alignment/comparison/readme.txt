MultiAlignCmd parameters

HDP
-d /home/joewandy/Dropbox/Project/documents/new_measure_experiment/input_data/P1/100 -gt /home/joewandy/Dropbox/Project/documents/new_measure_experiment/input_data/P1/ground_truth/ground_truth_100.txt -v -method myHdp -experimentType hdp -scoringMethod hdpmassrtjava -hdpLocalRtClusterStdev 2 -hdpGlobalRtClusterStdev 50 -hdpMassTol 500 -groupingNSamples 20 -groupingBurnIn 10

Precursor alignment
-d /home/joewandy/Dropbox/Project/documents/new_measure_experiment/input_data/P1/100 -gt /home/joewandy/Dropbox/Project/documents/new_measure_experiment/input_data/P1/ground_truth/ground_truth_100.txt -v -method myPrecursor -experimentType hdp -db /home/joewandy/git/metabolomics_tools/discretisation/database/std1_mols.csv -trans /home/joewandy/git/metabolomics_tools/discretisation/mulsubs/mulsub2.txt -v -binningMassTol 1000 -binningRtTol 5 -withinFileRtSd 2 -acrossFileRtSd 50

To check

770. avg m/z=978.681 avg RT=1914.26 prob=1.0
	feature_id   417 file_id 1 mz 978.68100 RT 1914.26 intensity 6.2726e+07	M+H@977.67372(1.00);bin_455_origin1

	PrecursorBin bin_id=455, len(features)=0, intensity=62725800.0, mass=977.673723548, mass_range=(976.6960498244613, 978.6513972715574), len(molecules)=0, origin=1, rt=1914.26, rt_range=(1909.26, 1919.26), top_id=395

Top 	PrecursorBin bin_id=395, len(features)=0, intensity=0.0, mass=977.673723548, mass_range=(976.6960498244613, 978.6513972715574), len(molecules)=0, origin=0, rt=0.0, rt_range=(0.0, 0.0)

=========================================

79. avg m/z=977.427 avg RT=1888.98 prob=1.0
	feature_id   394 file_id 0 mz 977.42700 RT 1888.98 intensity 5.9691e+07	M+H@976.41972(1.00);bin_575_origin0

	PrecursorBin bin_id=575, len(features)=0, intensity=59691200.0, mass=976.419723548, mass_range=(975.4433038244613, 977.3961432715573), len(molecules)=0, origin=0, rt=1888.98, rt_range=(1883.98, 1893.98), top_id=394

Top 	PrecursorBin bin_id=394, len(features)=0, intensity=0.0, mass=976.160723548, mass_range=(975.1845628244613, 977.1368842715573), len(molecules)=0, origin=0, rt=0.0, rt_range=(0.0, 0.0)

