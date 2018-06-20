function [theta_hat, Gamma, rho, kappa] = online_qSPICE(y_n, h_n, q, L, Gamma, rho, kappa, theta_hat, n)
P = length(h_n);

% Update online parameters
Gamma = Gamma + h_n * h_n';
rho = rho + h_n * y_n;
kappa = kappa + y_n' * y_n;

% Set instance parameters
nu = kappa + theta_hat' * Gamma * theta_hat - 2 * real(theta_hat' * rho);
eta = rho - Gamma * theta_hat;

% Is it really necessary to iterate through the entire grid? 
% By using a wideband dictionary as an intermediate step, we can choose to only update the bands with high power.

for k = 1:L
    for i = 1:P   
        alpha = nu + Gamma(i, i) * theta_hat(i)' * theta_hat(i) + 2 * real(theta_hat(i)' * eta(i));
        beta = Gamma(i, i);
        gamma = abs(eta(i) + Gamma(i, i) * theta_hat(i));

        % Estimate angle
        phi = angle(eta(i) + Gamma(i, i) * theta_hat(i)); 

        % Estimate range
        if sqrt(n^(1/q) - 1) * gamma > sqrt(alpha*beta - gamma^2)
            r = gamma / beta - (1 / beta) * sqrt((alpha * beta - gamma^2) / n^(1/q) - 1);
        else
            r = 0;
        end

        % Parameter estimate
        new_theta = r * exp(1i * phi);

        % Update instance parameters
        nu = nu + Gamma(i, i) * abs(theta_hat(i) - new_theta)^2 + 2 * real((theta_hat(i) - new_theta)' * eta(i));
        eta = eta + Gamma(:, i) * (theta_hat(i) - new_theta);

        theta_hat(i) = new_theta;
    end
end
end

