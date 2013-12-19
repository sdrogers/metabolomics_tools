clear all;
close all;
fname = '/Users/simon/git/metabolomics_tools/AdductLevels/std1pos.csv';
X = importdata(fname);

db = X.textdata(2:end,1);
iso = X.textdata(2:end,3);
adduct = X.textdata(2:end,2);
unique_db = unique(db);

for i = 1:length(unique_db)
    pos = find(strcmp(db,unique_db(i)));
    isotope = iso(pos)
    add = adduct(pos)
end