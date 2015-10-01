check_empty_rows <- function(df) {
    all_na_check <- apply(df, 1, function(x) { all(is.na(x)) })
    return(all_na_check)
}

remove_empty_rows <- function(df_list) {

    checks <- lapply(df_list, check_empty_rows)
    combined_check <- checks[[1]]
    for (i in 2:length(checks)) {
        combined_check <- combined_check & checks[[i]]
    }
    
    to_keep <- !combined_check
    processed_df_list <- lapply(df_list, function(x) { return(x[to_keep, ]) })
    return(processed_df_list)
    
}