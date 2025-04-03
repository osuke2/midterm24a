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
    Psi_no_const = cumsum(x, dPsi);

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