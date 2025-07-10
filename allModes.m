function allModes()
clc; clear; close all;

%% User Interface for Parameter Input
params = inputdlg({
    'Algorithm (RLS/LMS/NLMS):'; 
    'Mode (full/partial):';
    'Tonal frequencies [Hz] (comma-separated):';
    'RLS lambdas (comma-separated):';
    'LMS/NLMS mus (comma-separated):';
    'NLMS epsilons (comma-separated):';
    'Filter order:'
}, 'Parameter Tuning', [1 40], {
    'RLS', 'full', '1000,2725', '0.999,0.9999,0.99999', '0.001', '0.001', '8'
});

%% Parse User Inputs
algorithm = upper(params{1});
mode = lower(params{2});
tonalFreqs = str2num(params{3}); %#ok<ST2NM>
    lambda_values = str2num(params{4}); %#ok<ST2NM>
    mu_values = str2num(params{5}); %#ok<ST2NM>
    eps_values = str2num(params{6}); %#ok<ST2NM>
filterOrder = str2double(params{7});

%% Load Signals
fs = 44100;
s = load("clean_speech.txt");
noisySpeech = load("noisy_speech.txt");
w = load("external_noise.txt");
N = min([length(s), length(noisySpeech), length(w)]);

%% Initialize Results Storage
results = struct('Weights', {}, 'Params', {}, 'Legend', {});

%% Main Simulation Loop
switch algorithm
    case 'RLS'
        for lambda = lambda_values
            [~, weights] = runSimulation(filterOrder, algorithm, mode, tonalFreqs, lambda, 0, 0, fs, N, noisySpeech, w);
            
            % Create new figure for each lambda
            figure('Position', [100 100 1200 600], 'Color', 'w');
            hold on;
            grid on;
            colors = lines(filterOrder);
            
            % Plot all weights for current lambda
            for w_idx = 1:filterOrder
                plot(weights(w_idx,:), 'Color', colors(w_idx,:), 'LineWidth', 1.5, ...
                    'DisplayName', sprintf('w%d', w_idx));
            end
            
            title(sprintf('RLS Weight Evolution (λ=%.5f)', lambda));
            xlabel('Samples');
            ylabel('Weight Value');
            legend('show', 'Location', 'bestoutside');
            set(gca, 'FontSize', 12);
        end
        
    case {'LMS', 'NLMS'}
        for mu = mu_values
            if strcmp(algorithm, 'NLMS')
                for eps = eps_values
                    [~, weights] = runSimulation(filterOrder, algorithm, mode, tonalFreqs, 0, mu, eps, fs, N, noisySpeech, w);
                    
                    % Create new figure for each mu-epsilon combination
                    figure('Position', [100 100 1200 600], 'Color', 'w');
                    hold on;
                    grid on;
                    colors = lines(filterOrder);
                    
                    for w_idx = 1:filterOrder
                        plot(weights(w_idx,:), 'Color', colors(w_idx,:), 'LineWidth', 1.5, ...
                            'DisplayName', sprintf('w%d', w_idx));
                    end
                    
                    title(sprintf('NLMS Weight Evolution (μ=%.3f, ε=%.4f)', mu, eps));
                    xlabel('Samples');
                    ylabel('Weight Value');
                    legend('show', 'Location', 'bestoutside');
                    set(gca, 'FontSize', 12);
                end
            else
                [~, weights] = runSimulation(filterOrder, algorithm, mode, tonalFreqs, 0, mu, 0, fs, N, noisySpeech, w);
                
                % Create new figure for each mu
                figure('Position', [100 100 1200 600], 'Color', 'w');
                hold on;
                grid on;
                colors = lines(filterOrder);
                
                for w_idx = 1:filterOrder
                    plot(weights(w_idx,:), 'Color', colors(w_idx,:), 'LineWidth', 1.5, ...
                        'DisplayName', sprintf('w%d', w_idx));
                end
                
                title(sprintf('LMS Weight Evolution (μ=%.3f)', mu));
                xlabel('Samples');
                ylabel('Weight Value');
                legend('show', 'Location', 'bestoutside');
                set(gca, 'FontSize', 12);
            end
        end
end
end

%% Core Simulation Function
function [mse, weight_history] = runSimulation(filterOrder, algorithm, mode, tonalFreqs, lambda, mu, eps, fs, N, noisySpeech, w)
    % Initialize buffers
    buffer.W = zeros(filterOrder, 1);
    buffer.inputBuffer = zeros(filterOrder, 1);

    if strcmp(algorithm, 'RLS')
        buffer.P = 1000 * eye(filterOrder);
        buffer.lambda = lambda;
    end

    notchBuffer = initNotchBuffer(mode, tonalFreqs, fs, 0.9857);

    % Processing loop
    mse = zeros(N,1);
    weight_history = zeros(filterOrder, N);

    for n = 1:N
        w_n = w(n);
        x = noisySpeech(n);

        % Preprocessing
        if strcmp(mode, 'partial')
            [w_n, notchBuffer] = notchFilter(w_n, notchBuffer);
        end

        % Update buffer
        buffer.inputBuffer = [w_n; buffer.inputBuffer(1:end-1)];

        % Adaptive filtering
        if n >= filterOrder
            x_vec = buffer.inputBuffer;

            switch algorithm
                case 'RLS'
                    g = (buffer.P * x_vec) / (lambda + x_vec' * buffer.P * x_vec);
                    y = buffer.W' * x_vec;
                    e = x - y;
                    buffer.W = buffer.W + g * e;
                    buffer.P = (buffer.P - g * x_vec' * buffer.P) / lambda;

                case 'LMS'
                    y = buffer.W' * x_vec;
                    e = x - y;
                    buffer.W = buffer.W + mu * e * x_vec;

                case 'NLMS'
                    y = buffer.W' * x_vec;
                    e = x - y;
                    mu_eff = mu / (x_vec' * x_vec + eps);
                    buffer.W = buffer.W + mu_eff * e * x_vec;
            end

            % Store results
            mse(n) = e^2;
            weight_history(:,n) = buffer.W;
        end
    end
end

%% Notch Filter Functions
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

function notchBuffer = initNotchBuffer(mode, tonalFreqs, fs, r)
    notchBuffer = {};
    if strcmp(mode, 'partial')
        for k = 1:length(tonalFreqs)
            omega0 = 2 * pi * tonalFreqs(k) / fs;
            b = [1, -2*cos(omega0), 1];
            a = [1, -2*r*cos(omega0), r^2];
            notchBuffer{k} = struct('b', b, 'a', a, 'z', zeros(2,1));
        end
    end
end
