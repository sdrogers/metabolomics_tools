function [out,allAdducts] = getRatios(fileName,numeratorAdduct,denominatorAdduct)


X = importdata(fileName);
db = X.textdata(2:end,1);
iso = X.textdata(2:end,3);
adduct = X.textdata(2:end,2);
unique_db = unique(db);
allAdducts = unique(adduct);
out.databaseID = {};
out.data = [];
for i = 1:length(unique_db)
    pos1 = find(strcmp(db,unique_db(i)) & strcmp(adduct,numeratorAdduct));
    pos2 = find(strcmp(db,unique_db(i)) & strcmp(adduct,denominatorAdduct));
    if length(pos1)>0 && length(pos2)>0
        out.databaseID = [out.databaseID;unique_db(i)];
        out.data = [out.data;X.data(pos1,:)./X.data(pos2,:)];
    end
end