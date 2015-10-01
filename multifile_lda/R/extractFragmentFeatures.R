extract_ms2_fragment_df_single <- function(ms1, ms2, grouping_tol, common_bins) {
            
    # create an empty dataframe
    fragment_df <- data.frame(t(rep(NA,length(ms1$peakID))))
    fragment_df <- fragment_df[-1,] # remove first column
        
    ms1.names <- as.character(ms1$peakID) # set row names on ms1 dataframe
    ms2.names <- as.character(ms2$peakID) # set row names on ms2 dataframe

    copy_ms2 <- ms2        
    common_bins_len <- length(common_bins)        
    for (i in 1:length(common_bins)) {
        
        mz <- common_bins[i]
        print(paste(c("i=", i, "/", common_bins_len, 
                      ", fragment=", mz,
                      ", remaining=", nrow(copy_ms2)), collapse=""))
        
        word_found <- FALSE
        if (nrow(copy_ms2) > 0) {
        
            # calculate mz window
            max_ppm <- mz * grouping_tol * 1e-06
            
            # find peaks within that window
            match_idx <- which(sapply(copy_ms2$mz, function(x) {
                abs(mz - x) < max_ppm
            }))    
                    
            if (length(match_idx)>0) {
                word_found <- TRUE
            }
            
        }

        mean_mz <- round(mz, digits=5) # the bin label
        
        # if there's a match then use the actual fragment peaks
        if (word_found) { 
            
            # store the mean mz (bin id) into the original ms2 dataframe too
            peakids <- copy_ms2$peakID[match_idx]
            matching_pos <- match(as.character(peakids), ms2.names)
            ms2[matching_pos, "fragment_bin_id"] <- as.character(mean_mz)
            
            # get intensities
            intensities <- copy_ms2$intensity[match_idx]
            
            # get parent id
            parent_id <- copy_ms2$MSnParentPeakID[match_idx]
            
            # add a row of the intensities of the fragments
            parent_idx <- match(as.character(parent_id), ms1.names)
            row <- rep(NA, nrow(ms1))
            row[parent_idx] <- intensities                
            fragment_df <- rbind(fragment_df, row)
            rownames(fragment_df)[nrow(fragment_df)] <- paste(c("fragment_", mean_mz), collapse="")
            
            # remove fragments from ms2 list and start loop again with next fragment
            copy_ms2 <- copy_ms2[-match_idx,]
            
        } else { # otherwise just insert a row of all NAs   
            
            row <- rep(NA, nrow(ms1))
            fragment_df <- rbind(fragment_df, row)
            rownames(fragment_df)[nrow(fragment_df)] <- paste(c("fragment_", mean_mz), collapse="")
            
        }            
        
    }
    
    # all ms2 must be binned
    stopifnot(nrow(copy_ms2) == 0)  
    
    fragment_df <- fragment_df[mixedsort(row.names(fragment_df)),]
    names(fragment_df) <- paste(as.character(round(ms1$mz, digits=5)), 
                                as.character(ms1$rt),
                                as.character(ms1$peakID),
                                sep="_")
        
    # ms2 has been modified and needs to be returned too
    output <- list("fragment_df"=fragment_df, "ms2"=ms2)
    return(output)
    
}

get_common_fragments <- function(all_ms2, grouping_tol) {

    combined_ms2 <- do.call('rbind', all_ms2)
    combined_ms2 <- combined_ms2[with(combined_ms2, order(mz)), ]    
    common_bins <- vector()    
    while(nrow(combined_ms2) > 0) {
        
        # get first mz value
        mz <- combined_ms2$mz[1]
        print(paste(c("binning fragment=", mz, ", remaining=", nrow(combined_ms2)), collapse=""))                        
        
        # calculate mz window
        max_ppm <- mz * grouping_tol * 1e-06
        
        # find peaks within that window
        match_idx <- which(sapply(combined_ms2$mz, function(x) {
            abs(mz - x) < max_ppm
        }))    
        
        # calculate mean mz as label for ms2 row
        mean_mz <- mean(combined_ms2$mz[match_idx])
        common_bins <- c(common_bins, mean_mz)
                
        # remove fragments from ms2 list and start loop again with next fragment
        combined_ms2 <- combined_ms2[-match_idx,]
        
    }
    
    return(common_bins)
    
}

extract_ms2_fragment_df <- function(all_ms1, all_ms2, config) {

    stopifnot(length(all_ms1)==length(all_ms2))
    
    print("Constructing common fragment bins shared across files")
    
    grouping_tol <- config$MS1MS2_matrixGeneration_parameters$grouping_tol_frags  
    common_bins <- get_common_fragments(all_ms2, grouping_tol)
    
    fragment_matrices <- list()
    for (i in 1:length(all_ms2)) {

        print(paste(c("Constructing fragment matrix for file", i), collapse=" "))
        file_ms1 <- all_ms1[[i]]
        file_ms2 <- all_ms2[[i]]
        file_results <- extract_ms2_fragment_df_single(file_ms1, file_ms2, grouping_tol, common_bins)
        fragment_matrices[[i]] <- file_results$fragment_df
        all_ms2[[i]] <- file_results$ms2
        
    }
    all_results <- list("fragment_df"=fragment_matrices, "ms2"=all_ms2)
    return(all_results)
    
}