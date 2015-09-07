extract_ms2_fragment_df_single <- function(ms1, ms2, grouping_tol, common_bins) {
            
    # create an empty dataframe for existing words
    existing_fragment_df <- data.frame(t(rep(NA,length(ms1$peakID))))
    existing_fragment_df <- existing_fragment_df[-1,] # remove first column
    
    # create an empty dataframe for new words
    new_fragment_df <- data.frame(t(rep(NA,length(ms1$peakID))))
    new_fragment_df <- new_fragment_df[-1,] # remove first column
    
    ms1.names <- as.character(ms1$peakID) # set row names on ms1 dataframe
    ms2.names <- as.character(ms2$peakID) # set row names on ms2 dataframe

    copy_ms2 <- ms2        
    common_bins_len <- length(common_bins)        
    for (i in 1:length(common_bins)) {
        
        mz <- common_bins[i]
        print(paste(c("i=", i, "/", common_bins_len, 
                      ", mz=", mz,
                      ", remaining=", nrow(copy_ms2)), collapse=""))
        
        word_found <- FALSE
        if (nrow(copy_ms2) > 0) {
        
            # calculate mz window
            max.ppm <- mz * grouping_tol * 1e-06
            
            # find peaks within that window
            match.idx <- which(sapply(copy_ms2$mz, function(x) {
                abs(mz - x) < max.ppm
            }))    
                    
            if (length(match.idx)>0) {
                word_found <- TRUE
            }
            
        }

        mean.mz <- round(mz, digits=5) # the bin label
        
        # if there's a match then use the actual fragment peaks
        if (word_found) { 
            
            # store the mean mz (bin id) into the original ms2 dataframe too
            peakids <- copy_ms2$peakID[match.idx]
            matching_pos <- match(as.character(peakids), ms2.names)
            ms2[matching_pos, "fragment_bin_id"] <- as.character(mean.mz)
            
            # get intensities
            intensities <- copy_ms2$intensity[match.idx]
            
            # get parent id
            parent.id <- copy_ms2$MSnParentPeakID[match.idx]
            
            # add a row of the intensities of the fragments
            parent.idx <- match(as.character(parent.id), ms1.names)
            row <- rep(NA, nrow(ms1))
            row[parent.idx] <- intensities                
            existing_fragment_df <- rbind(existing_fragment_df, row)
            rownames(existing_fragment_df)[nrow(existing_fragment_df)] <- paste(c("fragment_", mean.mz), collapse="")
            
            # remove fragments from ms2 list and start loop again with next fragment
            copy_ms2 <- copy_ms2[-match.idx,]
            
        } else { # otherwise just insert a row of all NAs   
            
            row <- rep(NA, nrow(ms1))
            existing_fragment_df <- rbind(existing_fragment_df, row)
            rownames(existing_fragment_df)[nrow(existing_fragment_df)] <- paste(c("fragment_", mean.mz), collapse="")
            
        }            
        
    }
    
    # all ms2 must be binned
    stopifnot(nrow(copy_ms2) == 0)  
    
    existing_fragment_df <- existing_fragment_df[mixedsort(row.names(existing_fragment_df)),]
    names(existing_fragment_df) <- paste(as.character(round(ms1$mz, digits=5)), 
                                         as.character(ms1$rt),
                                         as.character(ms1$peakID),
                                         sep="_")
    
    new_fragment_df <- new_fragment_df[mixedsort(row.names(new_fragment_df)),]
    names(new_fragment_df) <- paste(as.character(round(ms1$mz, digits=5)), 
                                    as.character(ms1$rt),
                                    as.character(ms1$peakID),
                                    sep="_")
    
    fragment_df <- rbind(existing_fragment_df, new_fragment_df)    
    
    # ms2 has been modified and needs to be returned too
    output <- list("fragment_df"=fragment_df, "ms2"=ms2)
    return(output)
    
}

get_common_fragments <- function(combined_ms2, grouping_tol) {

    copy_ms2 <- combined_ms2    
    copy_ms2 <- copy_ms2[with(copy_ms2, order(mz)), ]    
    common_bins <- vector()    
    while(nrow(copy_ms2) > 0) {
        
        print(paste(c("binning=", nrow(copy_ms2)), collapse=""))
        
        # get first mz value
        mz <- copy_ms2$mz[1]
        
        # calculate mz window
        max.ppm <- mz * grouping_tol * 1e-06
        
        # find peaks within that window
        match.idx <- which(sapply(copy_ms2$mz, function(x) {
            abs(mz - x) < max.ppm
        }))    
        
        # calculate mean mz as label for ms2 row
        mean.mz <- mean(copy_ms2$mz[match.idx])
        common_bins <- c(common_bins, mean.mz)
                
        # remove fragments from ms2 list and start loop again with next fragment
        copy_ms2 <- copy_ms2[-match.idx,]
        
    }
    
    return(common_bins)
    
}

extract_ms2_fragment_df <- function(all_ms1, all_ms2, config) {

    stopifnot(length(all_ms1)==length(all_ms2))
    
    print("Constructing common fragment bins shared across files")
    
    grouping_tol <- config$MS1MS2_matrixGeneration_parameters$grouping_tol_frags  
    combined_ms2 <- do.call('rbind', all_ms2)
    common_bins <- get_common_fragments(combined_ms2, grouping_tol)
    
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