function conflict = actualConflict(m,L)
% actualConflict.m

% Description: compute the actual conflict for the three different 
% internal opinion vectors:
%                           s1 - the random one consists of -1 and 1;
%                           s2 - corresponds to the 10th samllest eigenvalue; 
% low-frequency on the graph;
%                           s3 - corresponds to the 10th largest eigenvalue;
% high-frequency on the graph

% Input:
%       m - represents the measure;
%       L - the laplacian matrix of the current network;

% Output:
%       conflict - the real conflict for the three internal opinion vectors
    
    n = size(L,1);
    I = eye(n);
    e = ones(n,1);
    J = e*e';
    conflict = zeros(1,3); 
    [~, ~, Mconf] = para4Measure(m, L);
    Mconf = (I-J/n)*Mconf*(I-J/n);
    s1= randn(n,1); 
    for i=1:n
        if s1(i) > 0
            s1(i) = 1;
        else
            s1(i) = -1;
        end
    end
    [vec,val] = eig(L);s2 = sign(vec(:,10));s3 = sign(vec(:,n-10)); 
    
    conflict(1) = s1'*Mconf*s1;
    conflict(2) = s2'*Mconf*s2;
    conflict(3) = s3'*Mconf*s3;
end