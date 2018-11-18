function badInd=getBadIndices(G)
Nvar=size(G,2);
badInd=false(Nvar,1);
for n=1:Nvar
    if ~any(G(:,n))
        badInd(n)=true;
    end
end