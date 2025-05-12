clc; clear; close all;

s = load("clean_speech.txt");
noisySpeech = load("noisy_speech.txt");
w = load("external_noise.txt");

fs = 44100;   
filterOrder = 5;                 % rls filter order
lambda = 0.9999;                 % Forgetting factor
r = 0.9857;                      % notch filter radius (BW = 200Hz Group delay=70 samples 1.588ms latency, Group delay < 5 for 500Hz on both sides )
mode = 'full';                   % mode options: partial / full

tonalFreqs = [];
if strcmp(mode, 'partial')
    a = inputdlg('Enter tonal frequencies:');
    tonalFreqs = str2num(a{1}); % user defined frequencies
end

N = length(noisySpeech);
v_hat = zeros(N, 1);
e = zeros(N, 1);

rlsBuffer.W = zeros(filterOrder, 1);
rlsBuffer.P = 1000 * eye(filterOrder);
rlsBuffer.lambda = lambda;
rlsBuffer.inputBuffer = zeros(filterOrder, 1);

notchBuffer = initNotchBuffer(mode, tonalFreqs, fs, r);

% ---real time emulation--- %
tic;

for n = 1:N
    w_n = w(n);
    x = noisySpeech(n);
    [v_hat(n), rlsBuffer, notchBuffer] = adaptiveNoiseFilter(x, w_n, rlsBuffer, notchBuffer, mode);
    
    e(n) = x - v_hat(n); 
    % e is heard by the user. In real implementation Speaker output = s - v_hat. 
    % x = s + v. So e = s + v - v_hat. 
end
toc;

audiowrite("recovered.wav", e / max(abs(e)) * 0.9, fs);
% analyzeRecoveredAudio(noisySpeech, w, e, s, fs, mode, tonalFreqs, r);
% uncomment this line to analyze the output

function [vHat, rlsBuffer, notchBuffer] = adaptiveNoiseFilter(x, w, rlsBuffer, notchBuffer, mode)
    if strcmp(mode, 'partial')
        [w, notchBuffer] = notchFilter(w, notchBuffer);
    end
    
    rlsBuffer.inputBuffer = [w; rlsBuffer.inputBuffer(1:end-1)];
    
    [vHat, rlsBuffer] = rlsFilter(x, rlsBuffer);
end

% ---filter definitions--- %
function [vHat, rlsBuffer] = rlsFilter(x, rlsBuffer)
    x_vec = rlsBuffer.inputBuffer;
    W = rlsBuffer.W;
    P = rlsBuffer.P;
    lambda = rlsBuffer.lambda;
    
    g = (P * x_vec) / (lambda + x_vec' * P * x_vec);
    vHat = W' * x_vec;
    W = W + g * (x - vHat);
    P = (P - g * x_vec' * P) / lambda;
    
    rlsBuffer.W = W;
    rlsBuffer.P = P;
end

function [y, notchBuffer] = notchFilter(x, notchBuffer)
    y = x;
    for k = 1:length(notchBuffer)
        b = notchBuffer{k}.b;
        a = notchBuffer{k}.a;
        z = notchBuffer{k}.z;
        y_n = b(1) * y + z(1);
        z(1) = b(2) * y - a(2) * y_n + z(2);
        z(2) = b(3) * y - a(3) * y_n;
        y = y_n;
        notchBuffer{k}.z = z;
    end
end

% Initialising the notch filter once the user inputs the frequency %
function notchBuffer = initNotchBuffer(mode, tonalFreqs, fs, r)
    if strcmp(mode, 'partial')
        for k = 1:length(tonalFreqs)
            omega0 = 2 * pi * tonalFreqs(k) / fs;
            b = [1, -2*cos(omega0), 1];
            a = [1, -2*r*cos(omega0), r^2];
            notchBuffer{k} = struct('b', b, 'a', a, 'z', zeros(2,1));
        end
    else
        notchBuffer = {};
    end
end

% ----------- Analysis -------------%
function analyzeRecoveredAudio(noisySpeech, w, recoveredSpeech, s, fs, mode, tonalFreqs, r)
    N = min([length(noisySpeech), length(w), length(recoveredSpeech), length(s)]);
    noisySpeech = noisySpeech(1:N);
    recoveredSpeech = recoveredSpeech(1:N);
    s = s(1:N);
    
    figure('Position', [100, 100, 800, 800]);
    
    subplot(3,1,1);
    spectrogram(noisySpeech, hamming(512), 256, 1024, fs, 'yaxis');
    title('Noisy Speech');
    clim([-80 -20]);
    
    subplot(3,1,2);
    spectrogram(recoveredSpeech, hamming(512), 256, 1024, fs, 'yaxis');
    title('Recovered Speech');
    clim([-80 -20]);
    
    subplot(3,1,3);
    [Pxx_orig, f] = pwelch(noisySpeech, hamming(1024), 512, 4096, fs);
    Pxx_rec = pwelch(recoveredSpeech, hamming(1024), 512, 4096, fs);
    Pxx_clean = pwelch(s, hamming(1024), 512, 4096, fs);
    
    plot(f, 10*log10(Pxx_orig));
    hold on;
    plot(f, 10*log10(Pxx_rec));
    plot(f, 10*log10(Pxx_clean));
    xlabel('Frequency (Hz)'); ylabel('Power (dB)');
    title('PSD Comparison'); grid on;
    legend('Noisy','Recovered','Clean');

    if strcmp(mode, 'partial')
        bandwidth = (1-r)*fs/pi;
        
        % Calculate suppression depth
        [H, f_resp] = freqz(recoveredSpeech, noisySpeech, 4096, fs);
        response_dB = 20*log10(abs(H));
        
        depth = zeros(size(tonalFreqs));
        for k = 1:length(tonalFreqs)
            idx = (f_resp >= tonalFreqs(k)-bandwidth/2) & (f_resp <= tonalFreqs(k)+bandwidth/2);
            depth(k) = mean(response_dB(idx));
        end
        
        stem(tonalFreqs, depth, 'filled', 'LineWidth', 2);
        hold on;
        yline(0, 'k--');
        xticks(tonalFreqs);
        xlabel('Frequency (Hz)');
        ylabel('Suppression (dB)');
        title(sprintf('Narrowband Suppression Depth (BW=%.1f Hz)', bandwidth));
        grid on;
        
        fprintf('\nNarrowband Suppression Depth (Bandwidth = %.1f Hz)\n', bandwidth);
        fprintf('--------------------------------------------\n');
        
        for k = 1:length(tonalFreqs)
            fprintf('Tone at %4d Hz: %5.1f dB suppression\n', tonalFreqs(k), depth(k));
        end
        
        % Calculate TNR
        [Pxx_noisy, f_tnr] = pwelch(noisySpeech, hamming(2048), 1024, 4096, fs);
        [Pxx_rec_tnr, ~] = pwelch(recoveredSpeech, hamming(2048), 1024, 4096, fs);

        TNR_before = zeros(size(tonalFreqs));
        TNR_after = zeros(size(tonalFreqs));

        fprintf('\nTonal-to-Noise Ratio (TNR):\n');
        fprintf('----------------------------\n');
        
        for k = 1:length(tonalFreqs)
            toneIdx = (f_tnr >= tonalFreqs(k) - bandwidth/2) & (f_tnr <= tonalFreqs(k) + bandwidth/2);
            noiseIdx = (f_tnr >= tonalFreqs(k) - 3*bandwidth) & (f_tnr < tonalFreqs(k) - 1.5*bandwidth) | ...
                       (f_tnr > tonalFreqs(k) + 1.5*bandwidth) & (f_tnr <= tonalFreqs(k) + 3*bandwidth);

            tonePower_before = mean(Pxx_noisy(toneIdx));
            noisePower_before = mean(Pxx_noisy(noiseIdx));
            TNR_before(k) = 10*log10(tonePower_before / noisePower_before);

            tonePower_after = mean(Pxx_rec_tnr(toneIdx));
            noisePower_after = mean(Pxx_rec_tnr(noiseIdx));
            TNR_after(k) = 10*log10(tonePower_after / noisePower_after);
            
            fprintf('Tone at %4d Hz: TNR Before = %5.1f dB, TNR After = %5.1f dB, ΔTNR = %5.1f dB\n', ...
                tonalFreqs(k), TNR_before(k), TNR_after(k), TNR_after(k) - TNR_before(k));
        end
    else        
        noisy_SNR = 10*log10(sum(s.^2) / sum((noisySpeech - s).^2));
        enhanced_SNR = 10*log10(sum(s.^2) / sum((recoveredSpeech - s).^2));
        
        fprintf('SNR Before Processing: %.2f dB\n', noisy_SNR);
        fprintf('SNR After Processing: %.2f dB\n', enhanced_SNR);
        fprintf('SNR Improvement: %.2f dB\n', enhanced_SNR - noisy_SNR);
    end
end