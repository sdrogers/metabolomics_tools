extract_ms2_loss_df_single <- function(ms1, ms2, grouping_tol, threshold_counts, common_bins) {
    
    # create an empty dataframe for existing words
    loss_df <- data.frame(t(rep(NA,length(ms1$peakID))))
    loss_df <- loss_df[-1,] # remove first column
    
    ms1.names <- as.character(ms1$peakID) # set row names on ms1 dataframe
    ms2.names <- as.character(ms2$peakID) # set row names on ms2 dataframe
    
    copy_ms2 <- ms2        
    common_bins_len <- length(common_bins)        
    for (i in 1:length(common_bins)) {
        
        mz <- common_bins[i]        
        word_found <- FALSE
        if (nrow(copy_ms2) > 0) {
            
            # calculate loss mz window
            max_ppm <- abs(mz*grouping_tol*1e-06)
            
            # find losses within that window
            temp <- abs(mz-copy_ms2$loss)
            match_idx <- which(temp <= abs(max_ppm))

            if (length(match_idx) > 0) {
                parent_id <- ms2$MSnParentPeakID[match_idx]
                parent_idx <- match(as.character(parent_id), ms1.names)
                if (length(parent_idx) >= threshold_counts) {
                    word_found <- TRUE
                } else {
                    word_found <- FALSE
                }
            } else {
                word_found <- FALSE
            }
            
        } else {
            break
        }
                
        mean_mz <- round(mz, digits=5) # the bin label
        
        # actually creates the matrix entries here
        if (word_found) { 

            print(paste(c("i=", i, "/", common_bins_len, 
                          ", loss=", mz,
                          ", remaining=", nrow(copy_ms2),
                          " accepted"), collapse="")) 
            
            # store the loss bin id into the original ms2 dataframe too
            peakids <- copy_ms2$peakID[match_idx]
            matching_pos <- match(as.character(peakids), ms2.names)
            ms2[matching_pos, "loss_bin_id"] <- as.character(mean_mz)
            
            # get intensities
            intensities <- copy_ms2$intensity[match_idx]
            
            # add a row of the intensities of the fragments
            row <- rep(NA, nrow(ms1))
            row[parent_idx] <- intensities                
            loss_df <- rbind(loss_df, row)
            rownames(loss_df)[nrow(loss_df)] <- paste(c("loss_", mean_mz), collapse="")
            
            # remove losses from ms2 list and start loop again with next loss value
            copy_ms2 <- copy_ms2[-match_idx,]
            
        } else { # otherwise just insert a row of all NAs   

            print(paste(c("i=", i, "/", common_bins_len, 
                          ", loss=", mz,
                          ", remaining=", nrow(copy_ms2),
                          " rejected"), collapse=""))            
            
            row <- rep(NA, nrow(ms1))
            loss_df <- rbind(loss_df, row)
            rownames(loss_df)[nrow(loss_df)] <- paste(c("loss_", mean_mz), collapse="")
            
        }            
        
    }
    
    # all ms2 must be binned
    # stopifnot(nrow(copy_ms2) == 0)  
    
    loss_df <- loss_df[mixedsort(row.names(loss_df)),]
    names(loss_df) <- paste(as.character(round(ms1$mz, digits=5)), 
                            as.character(ms1$rt),
                            as.character(ms1$peakID),
                            sep="_")
    
    # ms2 has been modified and needs to be returned too
    output <- list("loss_df"=loss_df, "ms2"=ms2)
    return(output)
    
}

get_common_losses <- function(all_ms1, all_ms2, grouping_tol, threshold_max_loss) {
        
    combined_ms2 <- do.call('rbind', all_ms2)
    combined_ms2 <- combined_ms2[with(combined_ms2, order(loss)), ]    
    common_bins <- vector()    
    while(nrow(combined_ms2) > 0) {
        
        # get first loss value
        mz <- combined_ms2$loss[1]
        print(paste(c("binning loss=", mz, ", remaining=", nrow(combined_ms2)), collapse=""))                
        if (mz > threshold_max_loss) {
            break
        }
        
        # calculate loss window
        # should use abs() here just in case there's MS2 mz > MS1 mz
        max_ppm <- abs(mz*grouping_tol*1e-06)
        
        # match to the first unmatched peak
        temp <- abs(mz-combined_ms2$loss)
        match_idx <- which(temp <= abs(max_ppm))
        
        # calculate mean loss mz as label for ms2 row
        mean_mz <- mean(combined_ms2$loss[match_idx])
        common_bins <- c(common_bins, mean_mz)
        
        # remove fragments from ms2 list and start loop again with next fragment
        combined_ms2 <- combined_ms2[-match_idx,]
        
    }
    
    return(common_bins)
    
}

extract_neutral_loss_df <- function(all_ms1, all_ms2, config) {
    
    stopifnot(length(all_ms1)==length(all_ms2))
    
    print("Constructing common loss bins shared across files")    

    # set a temporary loss column for the ms2 dataframe in each file
    for (i in 1:length(all_ms2)) {
        
        file_ms1 <- all_ms1[[i]]
        file_ms2 <- all_ms2[[i]] 
        
        file_ms1.names <- as.character(file_ms1$peakID) # set row names on ms1 dataframe
        file_ms2.names <- as.character(file_ms2$peakID) # set row names on ms2 dataframe
        
        # compute the difference between each fragment peak to its parent
        ms2_masses <- file_ms2$mz
        parent_ids <- file_ms2$MSnParentPeakID
        matches <- match(as.character(parent_ids), file_ms1.names)
        parent_masses <- file_ms1[matches, 5] # column 5 is the mz
        losses <- parent_masses - ms2_masses
        
        file_ms2[, "loss"] <- losses
        all_ms2[[i]] <- file_ms2
        
    }
 
    # make the common loss bins across files
    grouping_tol <- config$MS1MS2_matrixGeneration_parameters$grouping_tol_losses
    threshold_max_loss <- config$MS1MS2_matrixGeneration_parameters$threshold_max_loss    
    threshold_counts <- config$MS1MS2_matrixGeneration_parameters$threshold_counts    
    common_bins <- get_common_losses(all_ms1, all_ms2, grouping_tol, threshold_max_loss)
    
    # individually group each file according to the common loss bins
    loss_matrices <- list()
    for (i in 1:length(all_ms2)) {
        
        print(paste(c("Constructing loss matrix for file", i), collapse=" "))
        file_ms1 <- all_ms1[[i]]
        file_ms2 <- all_ms2[[i]]
        file_results <- extract_ms2_loss_df_single(file_ms1, file_ms2, grouping_tol, threshold_counts, common_bins)
        loss_matrices[[i]] <- file_results$loss_df
        
        # remove temporary loss column from the final results
        file_results$ms2$loss <- NULL
        all_ms2[[i]] <- file_results$ms2
        
    }
    all_results <- list("loss_df"=loss_matrices, "ms2"=all_ms2)
    return(all_results)
    
}