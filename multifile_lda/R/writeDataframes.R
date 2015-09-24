write_output <- function(ms1_list, ms2_list, fragment_df_list, loss_df_list, config) {
    
    for (i in 1:length(ms1_list)) {

        # construct the output filenames
        prefix <- config$input_files$prefix
        fragments_out <- paste(c(prefix, '_fragments_', i, '.csv'), collapse="")
        losses_out <- paste(c(prefix, '_losses_', i, '.csv'), collapse="")
        ms1_out <- paste(c(prefix, '_ms1_', i, '.csv'), collapse="")
        ms2_out <- paste(c(prefix, '_ms2_', i, '.csv'), collapse="")
        
        # write stuff out
        write.table(ms1_list[[i]], file=ms1_out, col.names=NA, row.names=T, sep=",")
        write.table(ms2_list[[i]], file=ms2_out, col.names=NA, row.names=T, sep=",")    
        write.table(fragment_df_list[[i]], file=fragments_out, col.names=NA, row.names=T, sep=",")
        write.table(loss_df_list[[i]], file=losses_out, col.names=NA, row.names=T, sep=",")
        
    }
    
}