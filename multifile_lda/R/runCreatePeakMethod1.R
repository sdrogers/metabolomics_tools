### This is the initial peak detection workflow we have from before --- using Tony's script ###

source('xcmsSetFragments.R')

run_create_peak_method_1 <- function(config) {
  
    input_dir <- config$input_files$fragmentation_file_m1
    files <- list.files(path=input_dir, pattern="*.mzXML", full.names=T, recursive=FALSE)
    results <- lapply(files, function(input_file) {
        
        print(paste(c("Processing ", input_file), collapse=""))

        # do peak detection using CentWave
        xset <- xcmsSet(files=input_file, method=config$MS1_XCMS_peakpicking_settings$method, ppm=config$MS1_XCMS_peakpicking_settings$ppm, snthresh=config$MS1_XCMS_peakpicking_settings$snthres, peakwidth=c(config$MS1_XCMS_peakpicking_settings$peakwidth_from,config$MS1_XCMS_peakpicking_settings$peakwidth_to),
                        prefilter=c(config$MS1_XCMS_peakpicking_settings$prefilter_from,config$MS1_XCMS_peakpicking_settings$prefilter_to), mzdiff=config$MS1_XCMS_peakpicking_settings$mzdiff, integrate=config$MS1_XCMS_peakpicking_settings$integrate, fitgauss=config$MS1_XCMS_peakpicking_settings$fitgauss, verbose.column=config$MS1_XCMS_peakpicking_settings$verbose.column)
        xset <- group(xset)
        
        # run Tony's script
        frags <- xcmsSetFragments(xset, cdf.corrected = config$ms1_ms2_pairing_parameters$cdf.corrected, min.rel.int=config$filtering_parameters_MassSpectrometry_related$min.rel.int, max.frags = config$filtering_parameters_MassSpectrometry_related$mass.frags, 
                                  msnSelect=c(config$ms1_ms2_pairing_parameters$msnSelect), specFilter=c(config$ms1_ms2_pairing_parameters$specFilter), match.ppm=config$ms1_ms2_pairing_parameters$match.ppm, 
                                  sn=config$filtering_parameters_MassSpectrometry_related$sn, mzgap=config$filtering_parameters_MassSpectrometry_related$mz_gap, min.r=config$ms1_ms2_pairing_parameters$min.r, min.diff=config$ms1_ms2_pairing_parameters$min.diff)
        peaks <- as.data.frame(frags@peaks)
        
        return(peaks)

    })
    return(results)
    
}