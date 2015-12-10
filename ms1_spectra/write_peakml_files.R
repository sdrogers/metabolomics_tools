# function to check if the package specified is present
# if yes then load it, otherwise install and load it
loadOrInstall <- function(package) {
    if (require(package, character.only=TRUE, quietly=TRUE)) {
        print(paste(package, " loaded"))
    } else {
        print(paste("Installing ", package))
        install.packages(package, dep=TRUE)
        if (require(package, character.only=TRUE, quietly=TRUE)) {
            print(paste(package, " installed and loaded"))
        } else {
            stop(paste("Could not install ", package))
        }
    }
}

# prompt user for the current working directory
loadOrInstall("tcltk")
workingDir <- tclvalue(tkchooseDirectory())
setwd(workingDir)

# function to process a single mzxml file
# vals is the parameters specified by user in GUI
processFile <- function(mzXMLfile, params) {

    # initialize rJava and mzmatch.R
    library(rJava)
    # if (.Platform$OS.type == 'windows' & .Platform$r_arch == 'i386') {
        print('Initialising JVM')
        .jinit()
    # }
    library(mzmatch.R)
    mzmatch.init(4000)

    library(xcms)

    # set output file
	outputFile <- sub(".mzXML", ".peakml", mzXMLfile)
	cat(sprintf("Input %s output %s\n", mzXMLfile, outputFile))
    
    # extract xcms parameters
    ppm <- as.numeric(params[1])
    peakWidthFrom <- as.numeric(params[2])
    peakWidthTo <- as.numeric(params[3])
    snthresh <- as.numeric(params[4])
    prefilterFrom <- as.numeric(params[5])
    prefilterTo <- as.numeric(params[6])
    integrate <- as.numeric(params[7])
    mzdiff <- as.numeric(params[8])
	fitgauss <- type.convert(params[9])

    # run xcms stuffs
	xset <- xcmsSet(mzXMLfile, method='centWave', 
		ppm=ppm,
		peakwidth=c(peakWidthFrom, peakWidthTo), 
		snthresh=snthresh,
		prefilter=c(prefilterFrom, prefilterTo), 
		integrate=integrate,
		mzdiff=mzdiff,
		fitgauss=fitgauss,
		verbose.columns=TRUE)
	PeakML.xcms.write.SingleMeasurement(xset=xset, filename=outputFile,
		ionisation="detect", addscans=2, writeRejected=FALSE, ApodisationFilter=TRUE)
		
}

# construct GUI
loadOrInstall("snow")
loadOrInstall("gWidgets")
loadOrInstall("gWidgetstcltk")
options(guiToolkit="tcltk")
fitgauss_options <- c("FALSE", "TRUE")
w <- gwindow("XCMSprocess-cluster", visible=FALSE)
g <- ggroup(cont=w, horizontal=FALSE)
lyt <- glayout(cont=g)
# ppm
lyt[1,1] <- "ppm:"
lyt[1,2] <- gedit("2", cont=lyt)
# peakwidth
lyt[2,1] <- "peakwidth: from "
lyt[2,2] <- gedit("10", cont=lyt)
lyt[2,3] <- " to "
lyt[2,4] <- gedit("100", cont=lyt)
# snthresh
lyt[3,1] <- "snthresh: "
lyt[3,2] <- gedit("5", cont=lyt)
# prefilter
lyt[4,1] <- "prefilter: from "
lyt[4,2] <- gedit("3", cont=lyt)
lyt[4,3] <- " to "
lyt[4,4] <- gedit("1000", cont=lyt)
# integrate
lyt[5,1] <- "integrate: "
lyt[5,2] <- gedit("1", cont=lyt)
# mzdiff
lyt[6,1] <- "mzdiff: "
lyt[6,2] <- gedit("0.001", cont=lyt)
# fitgauss
lyt[7,1] <- "fitgauss:"
lyt[7,2] <- gcombobox(fitgauss_options, cont=lyt)
# Number of processes to run
lyt[8,1] <- "# Processes:"
lyt[8,2] <- gedit("2", cont=lyt)
# action button
lyt[9,2] <- gbutton("Process", cont=lyt, handler=function(h,...) {
	params <- c(svalue(lyt[1,2]),   # ppm
	            svalue(lyt[2,2]),   # peakwidth from
	            svalue(lyt[2,4]),   # peakwidth to
	            svalue(lyt[3,2]),   # snthresh
	            svalue(lyt[4,2]),   # prefilter from
	            svalue(lyt[4,4]),   # prefilter to
	            svalue(lyt[5,2]),   # integrate
	            svalue(lyt[6,2]),   # mzdiff
	            svalue(lyt[7,2]))   # fitgauss
	visible(w) <- FALSE

	mzXMLfiles <- dir(full.names=TRUE, pattern="\\.mzXML$", recursive=TRUE)
	# lapply(mzXMLfiles, processFile, params)
    # run in parallel instead
	numProcesses <- svalue(lyt[8,2])
	cl <- makeCluster(rep("localhost", numProcesses), type = "SOCK", outfile="")
	clusterApply(cl, mzXMLfiles, processFile, params)

})
lyt[9,3] <- gbutton("Cancel", cont=lyt, handler = function(h,...) dispose(w))
visible(w) <- TRUE

