function y = get_complex_sinusoid(f, N, amp, dam)

if nargin < 1
    f = 0.1;
    N = 128;
    amp = 1;
    dam = 0;
elseif nargin < 2
    N = 128;
    amp = 1;
    dam = 0;
elseif nargin < 3 
    amp = 1;
    dam = 0;
elseif nargin < 4
    dam = 0;
end

y = amp*exp(1i*f*2*pi*(1:N)' + pi*rand*1i + dam*(1:N)');
end
