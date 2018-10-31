% Generate off-grid data SNR=10
N = 100;
f = [0.15, 0.31, 0.32];
amp = [0.2, 1, 1];
dam = [0, 0, 0];
sigma = 2.04/10;

y = zeros(N, 1);
for i = 1:length(f)
    y = y + get_complex_sinusoid(f(i), N, amp(i), dam(i));
end
y = y + sigma * (randn(N,1) + 1i * randn(N, 1)) / sqrt(2);

% Generate grid
P = 1024;
ff  = (0:P-1)/P-.5;   
A = get_fourier_matrix(N, ff);

% q-SPICE parameters
L = 2;
q = 1.5;

% Initialize variables
theta_hat = zeros(P,1);
Gamma = zeros(P,P);
rho   = zeros(P,1);
kappa = zeros(1,1);

% Run q-SPICE
for n = 1:N
    y_n = y(n);
    h_n = A(n, :); h_n = h_n';

    [theta_hat, Gamma, rho, kappa] = online_qSPICE(y_n, h_n, q, L, Gamma, rho, kappa, theta_hat, n);
end

figure()
plot(ff, abs(theta_hat).^2)
xlabel('Normalized frequency')
title('Final estimate')
