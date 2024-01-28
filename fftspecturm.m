function [f,P1] = fftspecturm(X,Fs,display)
    L = length(X);
    plottype = 'nonlog';
    Y = fft(X);
    P2 = abs(Y/L);
    P1 = P2(1:floor(L/2)+1);
    P1(2:end-1) = 2*P1(2:end-1);
    f = [Fs*(0:(L/2))/L]';

if display
    % figure;
    % plot(f,20*log10(P1));
    switch plottype
        case 'nonlog'
            plot(f,P1);
%             hold on;
        case 'logy'
            semilogy(f,P1);
    end
    % title('Single-Sided Amplitude Spectrum of X(t)')
    xlabel('Freq (Hz)')%{\itf}
    ylabel('|P({\itf})|')
end
end