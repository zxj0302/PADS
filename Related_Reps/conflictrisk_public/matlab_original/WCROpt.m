function [OptA, acr, wcr, conflicts] = WCROpt(A, iter, k, dim, min_eig)
% cvx_precision best

L = diag(sum(A)) - A;
n = size(A,1);
I = eye(n);
e = ones(n,1);
J = e*e'; % J is the all one matrix
[acr0,~,~] = para4Measure(3, L); % risk for measure m at for the original network
conflict0 = actualConflict(3,L); % for the three internal opinion vectors

% for WCR optimization
values = []; % stores the worst case conflict values Tr(s'*M*s) in the second column

disp('WCR/ACR optimization started');
for i = 1:(iter+1)
    [~, ~, M] = para4Measure(3, L); % get the middle matrix for measure m
    M = (I-J/n)*M*(I-J/n); 
    cvx_begin
        cvx_solver mosek
        cvx_precision best
        variable X(n,n) symmetric
        maximize sum(sum(X.*M))
        subject to
            diag(X) == 1
            X == semidefinite(n)
    cvx_end

    eig_min = min(eig(X));
    if eig_min < 0
        disp(['Matrix X is not positive definite. Minimum eigenvalue: ', num2str(eig_min)]);
        % Add a small positive value to the diagonal to make it positive definite
        X = X + (min_eig - eig_min) * eye(n);
        % Renormalize the diagonal entries to 1
        d = diag(X);
        X = X ./ (d * ones(1, n));
        disp(['Matrix X fixed to be positive definite with min eigenvalue: ', num2str(min(eig(X)))]);
    end

    C = chol(X)';
    V = sign(C*randn(size(C,2),dim)); % relaxation
    values = [values; cvx_optval max(diag(V'*M*V))];
    disp('values computed');

    if i == (iter+1)
        % If it's the last iteration, we need to jump out
        break;
    end
    V = (I-J/n)*V;

    disp('Using gradient descent');
    WGm = worstCaseRiskGradient(3, L, V);
    cvx_begin
        cvx_solver mosek
        % cvx_precision best
        variable stepMatrix(n,n) symmetric
        maximize sum(sum(WGm.*(diag(sum(stepMatrix))-stepMatrix)))
        subject to
            diag(stepMatrix) == zeros(n,1)
            A-1*stepMatrix >= 0
            A-1*stepMatrix <= 1
            sum(sum(abs(stepMatrix))) <= k
    cvx_end
    A = A-1*stepMatrix;
    disp('One step done');
    
    L = diag(sum(A)) - A;
    [acrs(i), ~, ~] = para4Measure(3, L);
    % actual conflict for the three internal opinion vectors
    conflicts(i,:) = actualConflict(3,L);
    imagesc(A),colorbar
    pause(0.01)
end

disp('All iterations finished');
OptA = A;
acr = [acr0;acrs'];
wcr = values(:,2);
conflicts = [conflict0;conflicts];
end
