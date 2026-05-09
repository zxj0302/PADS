function WGm = worstCaseRiskGradient(m, L, S)
% worstCaseRiskGradient.m

% Description: used for getting the gradients for the WCR for current
% network L w.r.t different conflict measures

% Input:
%       m - represents the measure;
%       L - the laplacian matrix of the current network;
%       S - a set of worst case internal opinions (size = dim) for the 
% current network, which is the result of the max-cut problem in the previous step.

% Output:
%       Gm    - the Gradient of the WCR for current network;

    n = size(L,2);
    I = eye(n);
    if m == 1
        WGm = L*inv(L+I)^2*S*S'*inv(L+I) + inv(L+I)*S*S'*inv(L+I)^2*L;
    elseif m == 2
        WGm = inv(L+I)^2*S*S'*inv(L+I)^2 - L*inv(L+I)^2*S*S'*inv(L+I)^2*L;
    elseif m == 3
        WGm = -inv(L+I)*S*S'*inv(L+I)^2 - inv(L+I)^2*S*S'*inv(L+I);
    else
        WGm = -inv(L+I)*S*S'*inv(L+I);
    end
end