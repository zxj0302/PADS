function [OptA, acr, wcr, conflicts] = ConflictRiskOptimization(A, ms, grad, avgCase, iter, k, stepsz,dim, min_eig)
cvx_precision best
% ConflictRiskOptimization.m 

% Description: 
% Given the adjacency matrix of a social network, this code optimizes 
% the average-case conflict risk (ACR) or the worst-case conflict risk (WCR)
% with respect to different conflict measures 
% using projected gradient descent or coordinate descent

% Input:
%       A       - the adjacency matrix of the network to be optimized
%       ms      - the measure for which the conflict risk is optimized
%       grad    - projected gradient descent or coordinate descent
%       avgCase - optimize for the average-case internal opinion vector or
% the worst case s
%       iter    - the iteration times
%       k       - upper bounds for total wieght changes that can be made at 
% every iteration, 2 for budget 1 since the A is symmetric
%       stepsz  - stepsize for coordinate descent, usually set it to 1, 
% but 0<=stepsz<=1 is also feasible
%       dim     - number of worst case opinions, default value is 10


% Output:
%       OptA        - the adjacency matrix of the optimized network
%       acr         - average-case conflict risks in the process of optimization
%       wcr         - worst-case conflict risks in the process of optimization
%       conflicts   - the actual risks for the 3 internal opinion vectors


%% current network input

L = diag(sum(A)) - A;
n = size(A,1);
I = eye(n);
e = ones(n,1);
J = e*e'; % J is the all one matrix
[acr0,~,~] = para4Measure(ms, L); % risk for measure m at for the original network
conflict0 = actualConflict(ms,L); % for the three internal opinion vectors

% for WCR optimization
values = []; % stores the worst case conflict values Tr(s'*M*s) in the second column

disp('WCR/ACR optimization started');
for i = 1:iter
    if avgCase
        % Optimize the ACR
        
        % 1. find the current WCR when optimizing the ACR
        [~, ~, M] = para4Measure(ms, L);
        M = (I-J/n)*M*(I-J/n); % to make the opinion vecter have 0 mean
        cvx_begin
            variable X(n,n) symmetric
            maximize sum(sum(X.*M))
            subject to
                diag(X) == 1
                X - min_eig * eye(n) == semidefinite(n)
        cvx_end
        C = chol(X)';
        V = sign(C*randn(size(C,2),dim)); % relaxation
        values = [values; cvx_optval max(diag(V'*M*V))];
        disp('values computed');
        
        [~, AGm, ~] = para4Measure(ms, L);
        % 2. find the step to take
        if grad
            % Optimize ACR using projected gradient descent
            cvx_begin
                variable stepMatrix(n,n) symmetric
                maximize sum(sum((diag(sum(stepMatrix)) - stepMatrix).*AGm))
                subject to 
                    diag(stepMatrix) == zeros(n,1)
                    A-1*stepMatrix >= 0
                    A-1*stepMatrix <= 1
                    sum(sum(abs(stepMatrix))) <= k 
            cvx_end
            % in case all the edges are deleted - then stop
            if sum(sum(round(A - stepMatrix, 5))) == 0
                break;
            end
            A = A - 1*stepMatrix; % network update
        else
            % Optimize ACR using coordinate descent
            Incr = 1000*ones(size(A));
            Decr = 1000*ones(size(A));
            for ii=1:1:n
                for jj = ii+1:1:n
                    delta = AGm(ii,ii) + AGm(jj,jj) - AGm(ii,jj) - AGm(jj,ii);
                    Decr(ii,jj) = -delta;
                    Incr(ii,jj) = delta;
                    % ensure the weights are within the range [0,1]
                    if A(ii,jj) > 1-stepsz
                        Incr(ii,jj) = 1000;
                    elseif A(ii,jj) < stepsz
                        Decr(ii,jj) = 1000;
                    end
                end
            end
            if min(min(Decr)) < min(min(Incr)) && min(min(Decr)) <= 0
                [imin, jmin] = find(Decr == min(min(Decr)));
                if size(imin,1) > 1
                    imin = imin(1);jmin = jmin(1);
                end
                A(imin, jmin) = A(imin, jmin) - stepsz;
                A(jmin, imin) = A(imin, jmin);
            elseif min(min(Incr)) <= 0 || round(min(min(Incr)),15) == 0
                min(min(Incr))
                [imin, jmin] = find(Incr == min(min(Incr)));
                if size(imin,1) > 1
                    imin = imin(1);jmin = jmin(1);
                end
                A(imin, jmin) = A(imin, jmin) + stepsz;
                A(jmin, imin) = A(imin, jmin);
            else
                % if it stops improving the objective, then stop optimizing
                break;
            end
        end
        % update laplacina and store the current acr
        L = diag(sum(A)) - A;
        [acrs(i), ~, ~] = para4Measure(ms, L);
    else
        % Optimize the WCR
        [~, ~, M] = para4Measure(ms, L); % get the middle matrix for measure m
        % for worst case s, optimise the worst case tr(s'*M*s) over L
        % First solve maxcut problem to find worst-case binary opinion vector, 
        % or a set of approximately worst-case binary opinion vectors:
        M = (I-J/n)*M*(I-J/n); 
        cvx_begin
            variable X(n,n) symmetric
            maximize sum(sum(X.*M))
            subject to
                diag(X) == 1
                X - min_eig * eye(n) == semidefinite(n)
        cvx_end

        % verity whether the matrix is positive definite
        % if not, do some post-processing to make it positive definite
        % Check if X is positive definite and fix it if necessary
        eig_min = min(eig(X));
        if eig_min < min_eig
            disp(['Matrix X is not positive definite. Minimum eigenvalue: ', num2str(eig_min)]);
            % Add a small positive value to the diagonal to make it positive definite
            X = X + (min_eig - eig_min + 1e-8) * eye(n);
            % Renormalize the diagonal entries to 1
            d = diag(X);
            X = X ./ (d * ones(1, n));
            disp(['Matrix X fixed to be positive definite with min eigenvalue: ', num2str(min(eig(X)))]);
        end

        C = chol(X)';
        % C = cholcov(X, 0)';  % The 0 flag means it won't error on non-positive definite
        V = sign(C*randn(size(C,2),dim)); % relaxation
        values = [values; cvx_optval max(diag(V'*M*V))];
        disp('values computed');

        V = (I-J/n)*V;

        % then the minimization problem
        if grad
            disp('Using gradient descent');
            % Optimize WCR using projected gradient descent
            % WGm is the gradient of Tr(s'*M*s) for worst-case s
            WGm = worstCaseRiskGradient(ms, L, V);
            cvx_begin
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
        else
            % Optimize WCR using coordinate descent
            XX = worstCaseRiskGradient(ms, L, V);
            for k1=1:n
                for k2=1:n
                    XXnew(k1,k2) = XX(k1,k1)+XX(k2,k2)-XX(k1,k2)-XX(k2,k1);
                end
            end
            XX = XXnew;
            clear XXnew
            for k1=1:n
                for k2=1:n
                    % check 0<=A+step<=1
                    if XX(k1,k2)>0 & A(k1,k2)==0
                        XX(k1,k2)=0;
                    elseif XX(k1,k2)<0 & A(k1,k2)==1
                        XX(k1,k2)=0;
                    end
                end
            end

            [iii,jjj]=find(abs(triu(XX))==max(max(abs(triu(XX)))))
            if XX(iii(1),jjj(1))>0
%                 disp('delete')
                A(iii(1),jjj(1)) = 0;
                A(jjj(1),iii(1)) = 0;
            else
%                 disp('add')
                A(iii(1),jjj(1)) = 1;
                A(jjj(1),iii(1)) = 1;
            end
        end
        L = diag(sum(A)) - A;
        [acrs(i), ~, ~] = para4Measure(ms, L);
    end
    % actual conflict for the three internal opinion vectors
    conflicts(i,:) = actualConflict(ms,L);
    imagesc(A),colorbar
    pause(0.01)
end

% display some info
disp('All iterations finished');
[~, ~, M] = para4Measure(ms, L);
M = (I-J/n)*M*(I-J/n); 
cvx_begin
    variable X(n,n) symmetric
    maximize sum(sum(X.*M))
    subject to
        diag(X) == 1
        X - min_eig * eye(n) == semidefinite(n)
cvx_end
eig_min = min(eig(X));
if eig_min < min_eig
    disp(['Matrix X is not positive definite. Minimum eigenvalue: ', num2str(eig_min)]);
    % Add a small positive value to the diagonal to make it positive definite
    X = X + (min_eig - eig_min + 1e-8) * eye(n);
    % Renormalize the diagonal entries to 1
    d = diag(X);
    X = X ./ (d * ones(1, n));
    disp(['Matrix X fixed to be positive definite with min eigenvalue: ', num2str(min(eig(X)))]);
end
C = chol(X)';
V = sign(C*randn(size(C,2),dim)); % relaxation
values = [values; cvx_optval max(diag(V'*M*V))];
disp('Final wcr values computed');

OptA = A;
acr = [acr0;acrs'];
wcr = values(:,2);
conflicts = [conflict0;conflicts];
end
