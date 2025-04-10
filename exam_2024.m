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

% I set F(K)-Ψ(I,K)=log(AK^(1+α)) - log(IK). 


% Define parameters
params.A = 3;          % Risk aversion
params.a = 1.2;        % Capital share
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


% Define parameters
params.sigma = 0.4; % volatility
params.r = 0.02;    % Interest rate
params.d = 0;       % dividend
params.K = 10;      % Strike price
params.T = 1;       % End of the time
params.M = 1000;   % Grid size for stock
params.I = 100;     % Grid size for time    

% Solve the model
[v, t, ss] = BSPDE_FDM_fun(params);

% ts plane
[T, S] = meshgrid(t, ss);

% 3D plot
figure;
surf(T, S, v, 'EdgeColor', 'none'); 
colorbar;
xlabel('Time to Maturity T');
ylabel('Asset Price S');
zlabel('Option Value V(S,t)');
title('Black-Scholes PDE Solution for European Call Option');
view(135, 30);

figure('Position', [100, 100, 600, 400])
plot(S, v(:,1), 'LineWidth', 2)
grid on
set(gca, 'FontSize', 12)

hold on

y=max(S-params.K,0);
plot(S,y);

hold off

%*******************************************************************
%**                    Question E(Neural Networks)
%********************************************************************/

% I solved Question 1.

% Main script: Solving ODE using Neural Networks


%% Problem Definition
% ODE: 0.04Ψ(x) = -Ψ'(x)^(1/2) + 1 + Ψ'(x)(x^0.3 - 0.05x - (Ψ'(x))^(-1/2))
% Initial condition: Ψ(0.1) = -1.07515
% Domain: x ∈ [0.1, 10]

%% Initial condition
A = -1.07515;

%% Network Parameters
N = 5;          % Number of hidden neurons
learning_rate = 0.005;
max_its = 10000;
tol = 1e-5;

%% Initialize Network Parameters
% Random initialization with Xavier/Glorot initialization
rng(42); % For reproducibility
v = randn(N, 1) * sqrt(2/1);  % Input weights
theta = randn(N, 1) * sqrt(2/1);  % Hidden biases
w = randn(N, 1) * sqrt(2/N);  % Output weights

%% Generate Training Points
nx = 1000;
xgrid = linspace(0.1, 10, nx)';

%% Training Loop
tic;
fprintf('Training Neural Network...\n');
for it = 1:max_its
    % Forward Pass
    [Psi, dPsi] = forward_pass(xgrid, v, theta, w, A);
    
    % Compute Target (from RHS of ODE)
    target = 25 * dPsi .* (xgrid .^0.03 - 0.05 * xgrid) - 50 * sqrt(dPsi) + 25;
    
    % Compute Error
    error = mean((Psi - target).^2);
    
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
nx_test = 1000;
x_test = linspace(0.1, 10, nx_test)';
[Psi_nn, dPsi_nn] = forward_pass(x_test, v, theta, w, A);
c_nn = dPsi_nn.^(-0.5);

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
    A = params.A;
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
    tv = (log(A*K.^(a-1)) - log (d))/r;
    
    % Main iteration loop
    for n = 1:maxit
        v = tv;
        
        % Forward difference
        dVf(1:L-1) = diff(v)/dK;
        dVf(L) = (d*Kmax)^(-1);
        
        % Backward difference
        dVb(2:L) = diff(v)/dK;
        dVb(1) = (d*Kmin)^(-1);
        
        % Consumption and savings
        If = dVf.^(-1);
        muf = If - d*K;
        Ib = dVf.^(-1);
        mub = Ib - d*K;
        
        % Steady state values
        I0 = d.*K;
        dV0 = I0.^(-1);
        
        % Upwind scheme
        If2 = muf > 0;
        Ib2 = mub < 0;
        I02 = (1-If2-Ib2);
        dV_Upwind =max(dVf.*If2 + dVb.*Ib2 + dV0.*I02, 1e-8);
        
        % Update consumption and utility
        I = dV_Upwind.^(-1);
        u = log(A*K.^a)-log(I);
        
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


function [v, t, ss] = BSPDE_FDM_fun(params)
    % Extract parameters
    sigma = params.sigma;   
    r = params.r;      
    d = params.d;      
    K = params.K;       
    T = params.T;         
    M = params.M;      
    I = params.I;       

    % Setup stock grid
    smin = log(0.4);
    smax = log(1000);
    amin = exp(smin);
    amax = exp(smax);
    s = linspace(smin, smax, M)';
    ss = linspace(amin, amax, M)';
    ds = (smax - smin) / (M-1);
    
    % Set up time grid
    tmin = 0;
    tmax = T;
    t = linspace(tmin, tmax, I);
    dt = (tmax - tmin) / (I - 1);

    % Initialize v as a matrix to store results for all time steps
    v = zeros(M, I);
    v(:, I) = max(exp(s) -K, 0);  % Store initial condition at the last column (t = T)
    

    

    % Tridiagonal matrix coefficients
    sig2 = sigma*sigma;
    dss = ds*ds; 

    a = 0.5*dt*((r-sig2)/ds-sig2/dss);
    b = 1+dt*(r+sig2/dss);
    c = -0.5*dt*((r-sig2)/ds+sig2/dss);

    A = diag(b*ones(M-1,1)) + diag(a*ones(M-2,1), -1) + diag(c*ones(M-2,1), 1);

    % Implicit scheme
    for i = I-1:-1:1
        % RHS
        rhs = v(1:M-2,i+1);
        rhs(1) = v(1,i+1);
        rhs(M-1) = v(end,i+1) - c*(exp(smax) - K * exp(-r * (T - t(i))));


        % Solve the linear system
        v(1:M-1,i) = A \ rhs;
        v(M, i) = exp(smax) - K * exp(-r * (T - t(i)));
    end
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

   % Raw derivative
    g = Nx + x .* dNdx;

    % Ensure non-negativity
    dPsi = softplus(3*g)/3;

    % Integrate numerically
    Psi_no_const = cumtrapz(x, dPsi);

    % Adjust to satisfy Psi(0.1) = A
    [~, idx0] = min(abs(x - 0.1));
    Psi = Psi_no_const - Psi_no_const(idx0) + A;
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
        [Psi_plus, ~] = forward_pass(x, v_plus, theta, w, A);
        [Psi_minus, ~] = forward_pass(x, v_minus, theta, w, A);
        grad_v(i) = mean((Psi_plus - target).^2 - (Psi_minus - target).^2) / (2*epsilon);
    end
    
    % Gradient for theta
    for i = 1:N
        theta_plus = theta; theta_plus(i) = theta_plus(i) + epsilon;
        theta_minus = theta; theta_minus(i) = theta_minus(i) - epsilon;
        [Psi_plus, ~] = forward_pass(x, v, theta_plus, w, A);
        [Psi_minus, ~] = forward_pass(x, v, theta_minus, w, A);
        grad_theta(i) = mean((Psi_plus - target).^2 - (Psi_minus - target).^2) / (2*epsilon);
    end
    
    % Gradient for w
    for i = 1:N
        w_plus = w; w_plus(i) = w_plus(i) + epsilon;
        w_minus = w; w_minus(i) = w_minus(i) - epsilon;
        [Psi_plus, ~] = forward_pass(x, v, theta, w_plus, A);
        [Psi_minus, ~] = forward_pass(x, v, theta, w_minus, A);
        grad_w(i) = mean((Psi_plus - target).^2 - (Psi_minus - target).^2) / (2*epsilon);
    end
end

function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end

function dy = sigmoid_derivative(x)
    s = sigmoid(x);
    dy = s .* (1 - s);
end

function y = softplus(x)
    y = log(1 + exp(x));
end