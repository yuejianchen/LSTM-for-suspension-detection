function y = IFFTfilter(x,fs,fc)

L = length(x);
Y = fft(x);
Y(floor(floor(L/2)*fc/(fs/2)):floor(L/2)+1) = 0;
Y(floor(L/2)+1 : floor(L-floor(L/2)*fc/(fs/2))) = 0;
y = real(ifft(Y));
y = y(20:end-20);
end