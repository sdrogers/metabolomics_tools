library(mzmatch.R)
mzmatch.init (12000)
FILESf <- dir ('peakml',full.names=TRUE,pattern="\\.peakml$",recursive=TRUE)
FILESa <- dir ('peakml',full.names=FALSE,pattern="\\.peakml$",recursive=TRUE)
for(i in 1:length(FILESf)) {
  outputfile = paste('txt/',FILESa[i],'.txt',sep="")
  print(outputfile)
  annot = paste('relation.id','relation.ship')
  mzmatch.ipeak.convert.ConvertToText (i=FILESf[i],o=outputfile,v=T,annotations=annot)
}