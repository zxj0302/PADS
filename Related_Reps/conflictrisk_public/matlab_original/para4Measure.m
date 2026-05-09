function [riskM, AGm, M] = para4Measure(m, L)

% para4Measure.m

% Description: used for getting parameters for different m at different
% time step for average case internal opinion vector s

% Input:
%       m - represents the measure;
%       L - the laplacian matrix of the current network;

% Output:
%       riskM - current ACR w.r.t measure m;
%       Gm    - the Gradient of the ACR for current network;
%       M     - the current middle matrix for measuer m. 

    n = size(L,2);
    I = eye(n);
    if m == 1
        % internal conflict
        riskM = trace(inv(L+I)*L*L*inv(L+I));
        AGm = 2*inv(L+I)^2 - 2*inv(L+I)^3;
        M = inv(L+I)*L*L*inv(L+I); 
    elseif m == 2
        % external conflict
        riskM = trace(inv(L+I)*L*inv(L+I));
        AGm = -inv(L+I)^2 + 2*inv(L+I)^3;
        M = inv(L+I)*L*inv(L+I); 
    elseif m == 3
        % controversy
        riskM = trace(inv(L+I)*inv(L+I));
        AGm = -2*inv(L+I)^3;
        M = inv(L+I)*inv(L+I);
    else
        % resistance
        riskM = trace(inv(L+I));
        AGm = -inv(L+I)^2;
        M = inv(L+I);
    end
end