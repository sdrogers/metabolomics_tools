
# output_data = read.delim('mzMatchOutput.txt',sep='\t')
# final_data = which(colnames(output_data)=="relation.id")-1
# data_columns = 3:final_data
# average_spectrum = rowMeans(subset(output_data,select=final_data))
# output_data$peak.mean = average_spectrum

# # Select relation.id according to which cluster we want
# sub_output_data = output_data[output_data$relation.id == 0,] 

# # Normalise by max
# use_col = 'peak.mean'
# temp_col = sub_output_data[,use_col]
# temp_col = temp_col/max(temp_col)

# # Normalise by total
# use_col = 'peak.mean'
# temp_col = sub_output_data[,use_col]
# temp_col = temp_col / sum(temp_col)


# plot(c(min(sub_output_data$X),max(sub_output_data$X)),c(0,1),type='n')
# # plot.new()
# for (i in 1:nrow(sub_output_data)) {
#     lines(rep(sub_output_data[i,'X'],2),c(0,temp_col[i]))
#     }



# file_name = 'output.txt'
# write('BEGIN IONS',file_name,append=F)
# for (i in 1:nrow(sub_output_data)) {
#     line = paste(sub_output_data$X[i],temp_col[i],sep=" ")
#     write(line,file_name,append=T)
#     }
# write('END IONS',file_name,append=T)

make_mgf <- function(input_file,relation_id) {
    # Load the data
    output_data = read.delim(input_file,sep='\t')
    # Work out which columns have data in
    final_data = which(colnames(output_data)=="relation.id")-1
    data_columns = 3:final_data
    average_spectrum = rowMeans(subset(output_data,select=final_data))
    output_data$peak.mean = average_spectrum
    
    output_data$peak.mean.normalised = 100* average_spectrum / max(average_spectrum)
    
    # Select relation.id according to which cluster we want
    sub_output_data = output_data[output_data$relation.id == relation_id,] 
    
    output_file_name = 'output.mgf'
    write('BEGIN IONS',output_file_name,append=F)
    write('PEPMASS=0',output_file_name,append=T)
    for (i in 1:nrow(sub_output_data)) {
        line = paste(sub_output_data$X[i],output_data$peak.mean.normalised[i],sep=" ")
        write(line,output_file_name,append=T)
        }
    write('END IONS',output_file_name,append=T)
    }


make_msp <- function(input_file,output_file_name) {
    # Load the data
    output_data = read.delim(input_file,sep='\t')
    # Work out which columns have data in
    final_data = which(colnames(output_data)=="relation.id")-1
    data_columns = 3:final_data
    average_spectrum = rowMeans(subset(output_data,select=final_data))
    output_data$peak.mean = average_spectrum
    
    
    
    # Select relation.id according to which cluster we want
    # for(relation_id in 0:max(output_data$relation.id)) {
    for(relation_id in 0:2) {
        if(relation_id%%100 == 0) {
            print(paste('Exporting relation',relation_id))
        }
        sub_output_data = output_data[output_data$relation.id == relation_id,] 
        sub_output_data$peak.mean.normalised = 100* sub_output_data$peak.mean / max(sub_output_data$peak.mean)
        # write('BEGIN IONS',output_file_name,append=F)
        if(relation_id == 0) {
            write(paste('NAME: relation id',relation_id,sep=' '),output_file_name,append=F)    
        }else {
            write('',output_file_name,append=T)
            write(paste('NAME: relation id',relation_id,sep=' '),output_file_name,append=T)    
        }
        write(paste('DB#:',relation_id),output_file_name,append=T)
        write('Comments: nothing',output_file_name,append=T)
        write(paste('Num Peaks:',nrow(sub_output_data),sep=' '),output_file_name,append=T)
        # for (i in 1:nrow(sub_output_data)) {
        #     line = paste(sub_output_data$X[i],output_data$peak.mean.normalised[i],sep=" ")
        #     write(line,output_file_name,append=T)
        #     }
        B = matrix(c(sub_output_data$X,sub_output_data$peak.mean.normalised),nrow=nrow(sub_output_data),ncol=2)
        write.table(B,output_file_name,row.names=F,col.names=F,append=T)
        }
    }
# make_mgf('mzMATCHoutput.txt',0)
input_file = 'mzMATCHoutput.txt'
make_msp(input_file,'output.msp')
final_file_name = 'hitlist.tsv'
write(paste('NIST matches for',input_file),final_file_name,append=F)
write(paste('relation_id','name','formula','prob',sep='\t'),final_file_name,append=T)
a = system('C:\\2013_06_04_MSPepSearch_x32\\MSPepSearch.exe  M /HITS 10 /PATH C:\\NIST14\\MSSEARCH /MAIN mainlib /INP output.MSP /OUT test.txt /COL pz,cf')
b = readLines('test.txt')
for(i in 1:length(b)) {
    se = regexpr("relation id [0-9]*",b[i])
    if(se>-1) {
        se2 = regmatches(b[i],se)
        se2.split = strsplit(se2, ' ')[[1]]
        current_relation = se2.split[3]
        print(paste("Current relation: ",current_relation))
    }
    se = regexpr("Hit [0-9]*",b[i])
    if(se>-1) {
        name.start = regexpr("<<",b[i])
        name.end = regexpr(">>",b[i])+1
        name = substr(b[i],name.start,name.end) 
        remainder = substr(b[i],name.end+2,10000L)
        formula.start = regexpr("<<",remainder)
        formula.end = regexpr(">>",remainder) + 1
        formula = substr(remainder,formula.start,formula.end)
        pro = regexpr("Prob: [-+]?([0-9]*\\.[0-9]+|[0-9]+)",b[i])
        pro2 = strsplit(regmatches(b[i],pro),' ')[[1]]
        write(paste(current_relation,name,formula,pro2[2],sep='\t'),final_file_name,append=T)
        # dq = regexpr("<<(.*?)>>",b[i])
        # dq2 = regmatches(b[i],dq)
        # print(dq)
    }
}
