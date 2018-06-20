function A = get_fourier_matrix(N, ff)
A   = exp( 1i*2*pi*(1:N)'*ff );         % Fourier matrix.
end

