function timefreqspecturm(X,Fs,WindowLen)
    dt = 1/Fs;
    L = length(X);
    t = dt:dt:(L/Fs);

    figure;
    ax1 = subplot(3,3,[2 3]);
    plot(t,X);
    ylabel('Acc(m/s^2)');
    % axis([0 20 -3 3]);
    % axis([0 max(t_f) -0.3 0.3]);
    xlim([0 max(t)]);
    % ylim([-3 3]);

    ax2 = subplot(3,3,[4 7]);
    fftspecturm(X,Fs,1);
    % axis([0 3.2 0 0.015]);
    % axis([0 1.6 0 0.015]);
    % axis([0 Fs/2000 0 0.5]);
    xlim([0 Fs/2000]);
    view([-90 90]);

    ax3 = subplot(3,3,[5 6 8 9]);
    [s,f,t_temp]=spectrogram(X,hamming(Fs/WindowLen),round(Fs/(WindowLen+1)),[],Fs,'yaxis');
    [X,Y]=meshgrid(t_temp,f);   
    mesh(X,Y./1000,20*log10(abs(s)));
    hcb = colorbar;
    title(hcb,'Amplitude (dB)');
    colorbar off;
    caxis([-30 40]);
    view(2);
    % axis([0 20 0 3.2]);
    axis([0 max(t) 0 Fs/(2*1000)]);
    ylabel('Frequency(kHz)');
    xlabel('Time(s)');

    linkaxes([ax3 ax1], 'x');
    % linkaxes([ax3 ax2], 'y');
end
