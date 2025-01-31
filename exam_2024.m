%Advanced Macro Final exam
%Kei Hirano

clc; clear; close all;
%*******************************************************************
%**                Question3(MATLAB Implementation)
%********************************************************************/


% Define parameters
params.s = 2;          % Risk aversion
params.a = 0.3;        % Capital share
params.d = 0.05;       % Depreciation rate
params.r = 0.04;       % Interest rate
params.I = 1000;      % Grid size
params.maxit = 10000;  % Maximum iterations
params.crit = 1e-6;    % Convergence criterion
params.Delta = 1000;   % Time step

% Solve the model
[v, c, k, dist] = OneSecGrowth_FDM_fun(params);

% Value Function Plot
figure('Position', [100, 100, 900, 400])

subplot(1,2,1)
plot(k, v, 'LineWidth', 2)
grid on
set(gca, 'FontSize', 12)
xlabel('k')
ylabel('v(k)')
title('Value Function')

subplot(1,2,2)
plot(k, c, 'LineWidth', 2)
grid on
set(gca, 'FontSize', 12)
xlabel('k')
ylabel('c(k)')
title('Policy Function')


%*******************************************************************
%**            QuestionB-(5)(Consumption-Saving Problem)
%********************************************************************/


% Define parameters
params.s = 2;          % Risk aversion
params.o = 0.3;        % Constant difference
params.R = 0.05;       % Productivity
params.r = 0.04;       % Interest rate
params.I = 1000;      % Grid size
params.maxit = 10000;  % Maximum iterations
params.crit = 1e-6;    % Convergence criterion
params.Delta = 1000;   % Time step

% Solve the model
[v, c, a, dist] = Conssave_FDM_fun(params);

% Value Function Plot
figure('Position', [100, 100, 900, 400])

subplot(1,2,1)
plot(a, v, 'LineWidth', 2)
grid on
set(gca, 'FontSize', 12)
xlabel('a')
ylabel('v(a)')
title('Value Function')

subplot(1,2,2)
plot(a, c, 'LineWidth', 2)
grid on
set(gca, 'FontSize', 12)
xlabel('a')
ylabel('c(a)')
title('Policy Function')


%*******************************************************************
%**                    Question C(Investment Model)
%********************************************************************/

% I set F(K)-Ψ(I,K)=(K^αI)^(1-s)/(1-s). 


% Define parameters
params.s = 0.5;        % Risk aversion
params.a = 0.4;        % Capital share
params.d = 0.05;       % Depreciation rate
params.r = 0.04;       % Interest rate
params.L = 1000;       % Grid size
params.maxit = 10000;  % Maximum iterations
params.crit = 1e-6;    % Convergence criterion
params.Delta = 1000;   % Time step

% Solve the model
[v, I, K, dist] = Investment_FDM_fun(params);

% Value Function Plot
figure('Position', [100, 100, 900, 400])

subplot(1,2,1)
plot(K, v, 'LineWidth', 2)
grid on
set(gca, 'FontSize', 12)
xlabel('K')
ylabel('v(K)')
title('Value Function')

subplot(1,2,2)
plot(K, I, 'LineWidth', 2)
grid on
set(gca, 'FontSize', 12)
xlabel('K')
ylabel('I(K)')
title('Policy Function')

%*******************************************************************
%**                      Question D(BS-PDE)
%********************************************************************/


[v, t, s, ~] = BSPDE_FDM_fun();

% ts plane
[T, S] = meshgrid(t, s);

% 3D plot
figure;
surf(S, T, v, 'EdgeColor', 'none'); 
colorbar;
xlabel('Asset Price S');
ylabel('Time to Maturity T');
zlabel('Option Value V(S,t)');
title('Black-Scholes PDE Solution for European Call Option');
view(135, 30);

%*******************************************************************
%**                    Question E(Neural Networks)
%********************************************************************/

% I solved Question 1.

% Main script: Solving ODE using Neural Networks


%% Problem Definition
% ODE: 0.04Ψ(x) = -Ψ'(x)^(1/2) + Ψ'(x)(x^0.3 - 0.05x - (Ψ'(x))^(-1/2))
% Initial condition: Ψ(0.1) = -1.07515
% Domain: x ∈ [0, 2]

%% Initial condition
A = 0.1;

%% Network Parameters
N = 5;          % Number of hidden neurons
learning_rate = 0.1;
max_its = 10000;
tol = 1e-5;

%% Initialize Network Parameters
% Random initialization with Xavier/Glorot initialization
rng(42); % For reproducibility
v = randn(N, 1) * sqrt(2/1);  % Input weights
theta = randn(N, 1) * sqrt(2/1);  % Hidden biases
w = randn(N, 1) * sqrt(2/N);  % Output weights

%% Generate Training Points
nx = 50;
xgrid = linspace(0, 2, nx)';

%% Training Loop
tic;
fprintf('Training Neural Network...\n');
for it = 1:max_its
    % Forward Pass
    [Psi, dPsi] = forward_pass(xgrid, v, theta, w, A);
    
    % Compute Target (from RHS of ODE)
    target = -sqrt(dPsi) + dPsi .* (xgrid.^0.3 - 0.05*xgrid - dPsi.^(-0.5));
    target = 0.04 * Psi;
    
    % Compute Error
    error = mean((0.04*Psi - (-sqrt(dPsi) + dPsi .* (xgrid.^0.3 - 0.05*xgrid - dPsi.^(-0.5)))).^2);
    
    if mod(it, 1000) == 0
        fprintf('it %d: Error = %.6f\n', it, error);
    end
    
    if error < tol
        break;
    end
    
    % Backpropagation (using numerical gradients for simplicity)
    [grad_v, grad_theta, grad_w] = compute_gradients(xgrid, v, theta, w, target, A);
    
    % Update Parameters
    v = v - learning_rate * grad_v;
    theta = theta - learning_rate * grad_theta;
    w = w - learning_rate * grad_w;
end
toc;

%% Evaluate and Plot Results
nx_test = 50;
x_test = linspace(0.1, 10, nx_test)';
[Psi_nn, ~] = forward_pass(x_test, v, theta, w, A);
c_nn = Psi_nn.^(-2);

% Plotting
figure('Position', [100, 100, 900, 400]);
subplot(1,2,1)
plot(x_test, Psi_nn, 'b-', 'LineWidth', 2);
xlabel('k');
ylabel('v(k)');
title('Neural Network Solution');
grid on;

subplot(1,2,2)
plot(x_test, c_nn, 'b-', 'LineWidth', 2);
xlabel('k');
ylabel('c(k)');
title('Neural Network Solution');
grid on;
% ======================================
% Functions
% ======================================


function [v, c, k, dist] = OneSecGrowth_FDM_fun(params)
    % Solve continuous-time one sector growth model using finite difference method
    % Input: params struct with model parameters
    % Output: value function (v), consumption (c), capital grid (k), convergence path (dist)
    
    % Extract parameters
    s = params.s;
    a = params.a;
    d = params.d;
    r = params.r;
    I = params.I;
    maxit = params.maxit;
    crit = params.crit;
    Delta = params.Delta;
    
    
    
    % Setup capital grid
    kmin = 0.1;
    kmax = 10;
    k = linspace(kmin, kmax, I)';
    dk = (kmax-kmin)/(I-1);
    
    % Initialize arrays
    dVf = zeros(I,1);
    dVb = zeros(I,1);
    c = zeros(I,1);
    dist = zeros(maxit,1);
    
    % Initial guess for value function
    tv = (k.^a).^(1-s)/(1-s)/r;
    
    % Main iteration loop
    for n = 1:maxit
        v = tv;
        
        % Forward difference
        dVf(1:I-1) = diff(v)/dk;
        dVf(I) = (kmax^a - d*kmax)^(-s);
        
        % Backward difference
        dVb(2:I) = diff(v)/dk;
        dVb(1) = (kmin^a - d*kmin)^(-s);
        
        % Consumption and savings
        cf = dVf.^(-1/s);
        muf = k.^a - d.*k - cf;
        cb = dVb.^(-1/s);
        mub = k.^a - d.*k - cb;
        
        % Steady state values
        c0 = k.^a - d.*k;
        dV0 = c0.^(-s);
        
        % Upwind scheme
        If = muf > 0;
        Ib = mub < 0;
        I0 = (1-If-Ib);
        dV_Upwind = dVf.*If + dVb.*Ib + dV0.*I0;
        
        % Update consumption and utility
        c = dV_Upwind.^(-1/s);
        u = (c.^(1-s)-1)/(1-s);
        
        % Construct sparse transition matrix
        X = -min(mub,0)/dk;
        Y = -max(muf,0)/dk + min(mub,0)/dk;
        Z = max(muf,0)/dk;
        A = spdiags(Y,0,I,I) + spdiags(X(2:I),-1,I,I) + spdiags([0;Z(1:I-1)],1,I,I);
        
        % Check transition matrix
        if max(abs(sum(A,2))) > 1e-12
            error('Improper Transition Matrix');
        end
        
        % Solve system of equations
        B = (r + 1/Delta)*speye(I) - A;
        b = u + v/Delta;
        tv = B\b;
        
        % Check convergence
        Vchange = tv - v;
        dist(n) = max(abs(Vchange));
        
        if dist(n) < crit
            fprintf('Value Function Converged, Iteration = %d\n', n);
            dist = dist(1:n);
            break;
        end
    end
end


function [v, c, a, dist] = Conssave_FDM_fun(params)
    % Solve continuous-time one sector growth model using finite difference method
    % Input: params struct with model parameters
    % Output: value function (v), consumption (c), capital grid (a), convergence path (dist)
    
    % Extract parameters
    s = params.s;
    o = params.o;
    R = params.R;
    r = params.r;
    I = params.I;
    maxit = params.maxit;
    crit = params.crit;
    Delta = params.Delta;
    
    
    % Setup capital grid
    amin = 0.1;
    amax = 10;
    a = linspace(amin, amax, I)';
    da = (amax-amin)/(I-1);
    
    % Initialize arrays
    dVf = zeros(I,1);
    dVb = zeros(I,1);
    c = zeros(I,1);
    dist = zeros(maxit,1);
    
    % Initial guess for value function
    tv = (a).^(1-s)/(1-s)/r;
    
    % Main iteration loop
    for n = 1:maxit
        v = tv;
        
        % Forward difference
        dVf(1:I-1) = diff(v)/da;
        dVf(I) = (o + R*amax )^(-s);
        
        % Backward difference
        dVb(2:I) = diff(v)/da;
        dVb(1) = (o + R*amin)^(-s);
        
        % Consumption and savings
        cf = dVf.^(-1/s);
        muf = o + R*a - cf;
        cb = dVb.^(-1/s);
        mub = o + R*a - cb;
        
        % Steady state values
        c0 = o + R*a;
        dV0 = c0.^(-s);
        
        % Upwind scheme
        If = muf > 0;
        Ib = mub < 0;
        I0 = (1-If-Ib);
        dV_Upwind = dVf.*If + dVb.*Ib + dV0.*I0;
        
        % Update consumption and utility
        c = dV_Upwind.^(-1/s);
        u = (c.^(1-s)-1)/(1-s);
           % Construct sparse transition matrix
        X = -min(mub,0)/da;
        Y = -max(muf,0)/da + min(mub,0)/da;
        Z = max(muf,0)/da;
        A = spdiags(Y,0,I,I) + spdiags(X(2:I),-1,I,I) + spdiags([0;Z(1:I-1)],1,I,I);
        
        % Check transition matrix
        if max(abs(sum(A,2))) > 1e-12
            error('Improper Transition Matrix');
        end
        
        % Solve system of equations
        B = (r + 1/Delta)*speye(I) - A;
        b = u + v/Delta;
        tv = B\b;
        
        % Check convergence
        Vchange = tv - v;
        dist(n) = max(abs(Vchange));
        
        if dist(n) < crit
            fprintf('Value Function Converged, Iteration = %d\n', n);
            dist = dist(1:n);
            break;
        end
    end
end

     
function [v, I, K, dist] = Investment_FDM_fun(params)
    % Solve continuous-time one sector growth model using finite difference method
    % Input: params struct with model parameters
    % Output: value function (v), investment (I), capital grid (K), convergence path (dist)
    
    % Extract parameters
    s = params.s;
    a = params.a;
    d = params.d;
    r = params.r;
    L = params.L;
    maxit = params.maxit;
    crit = params.crit;
    Delta = params.Delta;

    
    % Setup capital grid
    Kmin = 0.1;
    Kmax = 10;
    K = linspace(Kmin, Kmax, L)';
    dK = (Kmax-Kmin)/(L-1);
    
    % Initialize arrays
    dVf = zeros(L,1);
    dVb = zeros(L,1);
    I = zeros(L,1);
    dist = zeros(maxit,1);
    
    % Initial guess for value function
    tv = (K).^(1-s)/(1-s)/r;
    
    % Main iteration loop
    for n = 1:maxit
        v = tv;
        
        % Forward difference
        dVf(1:L-1) = diff(v)/dK;
        dVf(L) = Kmax^a*(d*Kmax)^(-s);
        
        % Backward difference
        dVb(2:L) = diff(v)/dK;
        dVb(1) = Kmin^a*(d*Kmin)^(-s);
        
        % Consumption and savings
        If = (dVf.*K.^(-a)).^(-1/s);
        muf = If - d.*K;
        Ib = (dVb.*K.^(-a)).^(-1/s);
        mub = Ib - d.*K;
        
        % Steady state values
        I0 = d.*K;
        dV0 = K.^a.*I0.^(-s);
        
        % Upwind scheme
        If2 = muf > 0;
        Ib2 = mub < 0;
        I02 = (1-If2-Ib2);
        dV_Upwind =max(dVf.*If2 + dVb.*Ib2 + dV0.*I02, 1e-8);
        
        % Update consumption and utility
        I = (dV_Upwind.*K.^(-a)).^(-1/s);
        u = (K.^a.*I).^(1-s)/(1-s);
        
        % Construct sparse transition matrix
        X = -min(mub,0)/dK;
        Y = -max(muf,0)/dK + min(mub,0)/dK;
        Z = max(muf,0)/dK;
        P = spdiags(Y,0,L,L) + spdiags(X(2:L),-1,L,L) + spdiags([0;Z(1:L-1)],1,L,L);
        
        % Check transition matrix
        if max(abs(sum(P,2))) > 1e-12
            error('Improper Transition Matrix');
        end
        
        % Solve system of equations
        B = (r + 1/Delta)*speye(L) - P;
        b = u + v/Delta;
        tv = B\b;
        
        % Check convergence
        Vchange = tv - v;
        dist(n) = max(abs(Vchange));
        
        if dist(n) < crit
            fprintf('Value Function Converged, Iteration = %d\n', n);
            dist = dist(1:n);
            break;
        end
    end
end

function [v, t, s, dist] = BSPDE_FDM_fun()
    % Extract parameters
    sigma = 0.4;   
    r = 0.02;      
    d = 0.0;      
    K = 10;       
    T = 1;         
    M = 1000;      
    I = 100;       

    % Setup stock grid
    smin = 0.4;
    smax = 1000;
    s = linspace(smin, smax, M)';
    ds = (smax - smin) / (M - 1);
    
    % Set up time grid
    tmin = 0;
    tmax = T;
    t = linspace(tmin, tmax, I);
    dt = (tmax - tmin) / (I - 1);

    % Initial condition
    v = max(s - K, 0);

    % Initialize v as a matrix to store results for all time steps
    v_all = zeros(M, I);
    v_all(:, I) = v;  % Store initial condition at the last column (t = T)

    dist = zeros(I, 1);

    % Tridiagonal matrix coefficients
    j = (1:M-1)'; % j = 1, 2, ..., M-1
    a = 0.5 * (r - d) * j * dt - 0.5 * sigma^2 * j.^2 * dt;
    b = 1 + sigma^2 * j.^2 * dt + r * dt;
    c = -0.5 * (r - d) * j * dt - 0.5 * sigma^2 * j.^2 * dt;

    A = diag(b) + diag(a(2:M-1), -1) + diag(c(1:M-2), 1);

    % Implicit scheme
    for i = I-1:-1:1
        % RHS
        rhs = v(2:M);
        rhs(1) = rhs(1) - a(1) * v(1);
        rhs(end) = rhs(end) - c(M-1) * v(M);

        % Solve the linear system
        v(2:M) = A \ rhs;

        % Store the result for the current time step
        v_all(:, i) = v;

        % Check convergence
        if i < I-1
            dist(i) = max(abs(v - v_old));
            if dist(i) < 1e-6
                fprintf('Value Function Converged, Iteration = %d\n', i);
                break;
            end
        end
        v_old = v;
    end

    % Return the final v matrix
    v = v_all;
end


%% Helper Functions
function [Psi, dPsi] = forward_pass(x, v, theta, w, A)
    % Input dimensions:
    % x: [n_points × 1] - input values
    % v: [N × 1] - input weights
    % theta: [N × 1] - bias terms
    % w: [N × 1] - output weights
    % A: scalar - initial condition
    
    N = length(v);
    n_points = length(x);
    
    % Correct broadcasting for matrix operations
    x_broadcast = repmat(x, 1, N);      % [n_points × N]
    v_broadcast = repmat(v', n_points, 1);  % [n_points × N]
    theta_broadcast = repmat(theta', n_points, 1);  % [n_points × N]
    
    % Hidden layer computations
    z = v_broadcast .* x_broadcast + theta_broadcast;  % [n_points × N]
    y = sigmoid(z);  % [n_points × N]
    
    % Derivative computations
    dydx = v_broadcast .* sigmoid_derivative(z);  % [n_points × N]
    
    % Output layer
    Nx = y * w;  % [n_points × 1]
    dNdx = dydx * w;  % [n_points × 1]
    
    % Trial solution and its derivative
    Psi = A + x .* Nx;  % [n_points × 1]
    dPsi = Nx + x .* dNdx;  % [n_points × 1]
end
function [grad_v, grad_theta, grad_w] = compute_gradients(x, v, theta, w, target, A)
    % Compute gradients using numerical differentiation
    epsilon = 1e-6;
    N = length(v);
    grad_v = zeros(size(v));
    grad_theta = zeros(size(theta));
    grad_w = zeros(size(w));
    
    % Gradient for v
    for i = 1:N
        v_plus = v; v_plus(i) = v_plus(i) + epsilon;
        v_minus = v; v_minus(i) = v_minus(i) - epsilon;
        [~, dPsi_plus] = forward_pass(x, v_plus, theta, w, A);
        [~, dPsi_minus] = forward_pass(x, v_minus, theta, w, A);
        grad_v(i) = mean((dPsi_plus - target).^2 - (dPsi_minus - target).^2) / (2*epsilon);
    end
    
    % Gradient for theta
    for i = 1:N
        theta_plus = theta; theta_plus(i) = theta_plus(i) + epsilon;
        theta_minus = theta; theta_minus(i) = theta_minus(i) - epsilon;
        [~, dPsi_plus] = forward_pass(x, v, theta_plus, w, A);
        [~, dPsi_minus] = forward_pass(x, v, theta_minus, w, A);
        grad_theta(i) = mean((dPsi_plus - target).^2 - (dPsi_minus - target).^2) / (2*epsilon);
    end
    
    % Gradient for w
    for i = 1:N
        w_plus = w; w_plus(i) = w_plus(i) + epsilon;
        w_minus = w; w_minus(i) = w_minus(i) - epsilon;
        [~, dPsi_plus] = forward_pass(x, v, theta, w_plus, A);
        [~, dPsi_minus] = forward_pass(x, v, theta, w_minus, A);
        grad_w(i) = mean((dPsi_plus - target).^2 - (dPsi_minus - target).^2) / (2*epsilon);
    end
end

function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end

function dy = sigmoid_derivative(x)
    s = sigmoid(x);
    dy = s .* (1 - s);
end