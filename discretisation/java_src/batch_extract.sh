#! /bin/bash
for filename in /Users/simon/Dropbox/Bioresearch/Meta_clustering/StandardData140606/Positive/*.peakml; do
	echo "processing $filename"
	basefile="$(basename $filename).signal"
	echo "$basefile"
	java -ea -cp .:/Library/Frameworks/R.framework/Versions/3.1/Resources/library/mzmatch.R/java/mzmatch_2.0.jar GetSignals $filename /Users/simon/Dropbox/Bioresearch/Meta_clustering/StandardData140606/csv/$basefile
done