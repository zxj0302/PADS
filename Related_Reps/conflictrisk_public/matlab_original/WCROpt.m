function [OptA, acr, wcr, conflicts] = WCROpt(A, iter, k, dim, min_eig)
% cvx_precision best

L = diag(sum(A)) - A;
n = size(A,1);
I = eye(n);
e = ones(n,1);
J = e*e'; % J is the all one matrix
[acr0,~,~] = para4Measure(3, L); % risk for measure m at for the original network
conflict0 = actualConflict(3,L); % for the three internal opinion vectors

% Preflight: check CVX + Mosek compatibility. Some CVX/mosek combinations
% (older CVX with newer MOSEK 11) cause the cvx_mosek shim to reference
% a 'sol' field that Mosek no longer provides, producing the
% "Unrecognized field name 'sol'" error. If this occurs, instruct the
% user to update CVX (preferred) or install a Mosek version compatible
% with their CVX install.
try
    cvx_begin
        cvx_solver mosek
        cvx_precision best
        variable xx(2)
        minimize( norm(xx) )
        subject to
            xx >= 0
    cvx_end
catch ME
    if contains(ME.message, 'Unrecognized field name "sol"') || contains(ME.message, 'Unrecognized field name')
        error(['CVX/Mosek incompatibility detected: ', ME.message, '\n', ...
            'This typically happens when your CVX version is incompatible with your MOSEK installation.\n', ...
            'Fixes: (1) Update CVX to the latest release from http://cvxr.com/cvx and run `cvx_setup` in MATLAB;\n', ...
            '(2) Or install a MOSEK version compatible with your CVX (older MOSEK).\n', ...
            'After updating, restart MATLAB and try again.']);
    else
        rethrow(ME);
    end
end

% for WCR optimization
values = []; % stores the worst case conflict values Tr(s'*M*s) in the second column

disp('WCR/ACR optimization started');
for i = 1:(iter+1)
    [~, ~, M] = para4Measure(3, L); % get the middle matrix for measure m
    M = (I-J/n)*M*(I-J/n); 
    % Run the CVX SDP to compute X. Try Mosek first; on failure, retry
    % with SDPT3 and propagate a clear error if both fail.
    sdp_solved = false;
    try
        cvx_begin
            cvx_solver mosek
            cvx_precision best
            variable X(n,n) symmetric
            maximize sum(sum(X.*M))
            subject to
                diag(X) == 1
                X == semidefinite(n)
        cvx_end
        sdp_solved = true;
    catch ME
        warning('CVX with Mosek failed: %s', ME.message);
        try
            cvx_begin
                cvx_solver sdpt3
                cvx_precision best
                variable X(n,n) symmetric
                maximize sum(sum(X.*M))
                subject to
                    diag(X) == 1
                    X == semidefinite(n)
            cvx_end
            sdp_solved = true;
        catch ME2
            error('CVX failed with both Mosek and SDPT3: %s | %s', ME.message, ME2.message);
        end
    end

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
    % Solve the stepMatrix SDP. Try Mosek first; if it errors, retry
    % with SDPT3 and fail with an informative message if both fail.
    try
        cvx_begin
            cvx_solver mosek
            variable stepMatrix(n,n) symmetric
            maximize sum(sum(WGm.*(diag(sum(stepMatrix))-stepMatrix)))
            subject to
                diag(stepMatrix) == zeros(n,1)
                A-1*stepMatrix >= 0
                A-1*stepMatrix <= 1
                sum(sum(abs(stepMatrix))) <= k
        cvx_end
    catch ME
        warning('CVX stepMatrix with Mosek failed: %s', ME.message);
        try
            cvx_begin
                cvx_solver sdpt3
                variable stepMatrix(n,n) symmetric
                maximize sum(sum(WGm.*(diag(sum(stepMatrix))-stepMatrix)))
                subject to
                    diag(stepMatrix) == zeros(n,1)
                    A-1*stepMatrix >= 0
                    A-1*stepMatrix <= 1
                    sum(sum(abs(stepMatrix))) <= k
            cvx_end
        catch ME2
            error('CVX stepMatrix failed with both Mosek and SDPT3: %s | %s', ME.message, ME2.message);
        end
    end
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
