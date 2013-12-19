function out = makeMatrix(fname)

X = importdata(fname);

databaseID = unique(X.textdata(2:end,1));
adducts = unique(X.textdata(2:end,2));

nFiles = size(X.data,2);

out.matrix = zeros(length(databaseID),nFiles,length(adducts));
out.matrix = nan*out.matrix;
for i = 1:length(databaseID)
    for j = 1:length(adducts)
        pos = find(strcmp(X.textdata(2:end,1),databaseID{i}) & ...
            strcmp(X.textdata(2:end,2),adducts{j}));
        if ~isempty(pos)
            out.matrix(i,:,j) = X.data(pos,:);
        end
    end
end

out.adducts = adducts;
out.databaseID = databaseID;
out.files = X.textdata(1,4:end);
out.binarymatrix = ~isnan(out.matrix);


