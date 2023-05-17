function [center, neighbors] = findCenter(X1, X2 ,k)
    
    center = [mean(X1),mean(X2)]; 
    
    if length(X1) < k
        k = length(X1);
    end
    
    neighbors = zeros(k,2);
    Idx = knnsearch([X1,X2],[center(1),center(2)],'K',k);
    neighbors = [X1(Idx),X2(Idx)];
  
end