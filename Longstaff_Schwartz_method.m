%% Longstaff - Schwartz method (LSM) for Bermudan options' pricing

clear; clc;

S0 = 100;
K = 102;
r = 0.02; % We need r as we price under ℚ
sigma = 0.18;
T = 1; % The option has a (total) life of one year
T1 = 0.5;

q = 0; % By modifying the paid dividends we observe how much the values 
% get different between a Bermudan and an European (q > 0)

d1 = (log(S0/K) + (r + 0.5*sigma^2)*(T-T1))/(sigma*sqrt(T-T1));

d2 = (log(S0/K) + (r - 0.5*sigma^2)*(T-T1))/(sigma*sqrt(T-T1));
    % or d1 - sigma*sqrt(T-T1)
    
Nd1 = normcdf(d1);

Nd2 = normcdf(d2);

%% Case 1: Only one intermediary exercise date, T1 = 0.5 (6 months)
% Simulate M = 100,000 paths

clc;

M = 100000;

St1 = zeros(M,1);
St_final = zeros(M,1);

for i=1:M

    z1 = randn;
    z2 = randn;

    % We need two standardized gaussian random variables, as Brownian 
    % motions from where z descend have independent increment

    S1 = S0*exp((r - q - 0.5*sigma^2)*T1 + sqrt(T1)*sigma*z1);
    ST = S1*exp((r - q - 0.5*sigma^2)*(T-T1) + sqrt(T-T1)*sigma*z2);
    St1(i,1) = S1;
    St_final(i,1) = ST;

end

stock_path = [St1, St_final];

% Compute continuation value at T1 (intermediary date)
% Discount payoff of the European Call at T1 and check the moneyness

discounted_payoff = exp(-r*(T-T1))*max(stock_path(:,2)-K, 0);

ITM = stock_path(:,1) > K; % Boolean 0-1

% Keep only the ITM stock. It increases the algorithm efficiency and
% makes sense since if the option is OTM at T1 we have no incentive to
% exercise it

S1_itm = stock_path(ITM, 1);
discounted_payoff_itm  = discounted_payoff(ITM);

% We run an OLS regression to estimate continuation value at T1

iota = ones(size(S1_itm));

X_itm = [iota, S1_itm, S1_itm.^2];
beta = X_itm \ discounted_payoff_itm;
continuation_value_itm = X_itm * beta;

% Compare immediate payoff with continuation value
immediate_payoff_itm = S1_itm - K;
early_exercise_itm = immediate_payoff_itm > continuation_value_itm;

cashflow_t0 = exp(-r*T) * max(stock_path(:,2) - K, 0);

itm_indices = find(ITM);
exercise_indices = itm_indices(early_exercise_itm);
cashflow_t0(exercise_indices) = exp(-r*T1) * (stock_path(exercise_indices, 1) - K);

% Final price
bermudan_price = mean(cashflow_t0);
european_mc_price = mean(exp(-r*T) * max(stock_path(:,2) - K, 0));

fprintf('Bermudan Call Price (LSM): %.4f\n', bermudan_price);
fprintf('European Call Price (MC):  %.4f\n', european_mc_price);

% As expected, we have that, in absence of dividends, 
% the two prices cohincide, hence determining an exercise premium of 0. 

% If we set a dividend q, then if we take care of the downward bias
% we have Bermudan > European

%% Sensitivity plot: Bermudan vs European price as dividend yield q varies

rng(42);
z1_vec = randn(M,1);
z2_vec = randn(M,1);

q_vec = 0:0.005:0.10; % q varies from 0 to 10%
berm_vec = zeros(size(q_vec));
euro_vec = zeros(size(q_vec));

for k = 1:length(q_vec)
    q_curr = q_vec(k);
    
    % New stock path
    St1_q = S0*exp((r - q_curr - 0.5*sigma^2)*T1 + sqrt(T1)*sigma*z1_vec);
    St_final_q = St1_q.*exp((r - q_curr - 0.5*sigma^2)*(T-T1) + sqrt(T-T1)*sigma*z2_vec);
    stock_path_q = [St1_q, St_final_q];
    
    % Same as before
    discounted_payoff_q = exp(-r*(T-T1))*max(stock_path_q(:,2)-K, 0);
    ITM_q = stock_path_q(:,1) > K;
    
    S1_itm_q = stock_path_q(ITM_q, 1);
    discounted_payoff_itm_q = discounted_payoff_q(ITM_q);
    
    iota_q = ones(size(S1_itm_q));
    X_itm_q = [iota_q, S1_itm_q, S1_itm_q.^2, S1_itm_q.^3]; % Grado 3 riduce il downward bias
    beta_q = X_itm_q \ discounted_payoff_itm_q;
    continuation_value_itm_q = X_itm_q * beta_q;
    
    immediate_payoff_itm_q = S1_itm_q - K;
    early_exercise_itm_q = immediate_payoff_itm_q > continuation_value_itm_q;
    
    cashflow_t0_q = exp(-r*T) * max(stock_path_q(:,2) - K, 0);
    itm_indices_q = find(ITM_q);
    exercise_indices_q = itm_indices_q(early_exercise_itm_q);
    cashflow_t0_q(exercise_indices_q) = exp(-r*T1) * (stock_path_q(exercise_indices_q, 1) - K);
    
    berm_vec(k) = mean(cashflow_t0_q);
    euro_vec(k) = mean(exp(-r*T) * max(stock_path_q(:,2) - K, 0));
end

figure;
subplot(2,1,1);
plot(q_vec*100, berm_vec, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 4); hold on;
plot(q_vec*100, euro_vec, 'r--', 'LineWidth', 1.5);
xline(r*100, 'k:', 'q = r');
grid on;
xlabel('Dividend yield q (%)');
ylabel('Option Price');
title('Bermudan Call vs European Call as q varies');
legend('Bermudan Call (LSM)', 'European Call (MC)', 'Location', 'northeast');

subplot(2,1,2);
plot(q_vec*100, berm_vec - euro_vec, 'k-s', 'LineWidth', 1.5, 'MarkerSize', 4);
xline(r*100, 'k:', 'q = r');
yline(0, 'k-');
grid on;
xlabel('Dividend yield q (%)');
ylabel('Bermudan - European');
title('Early Exercise Premium as q varies');

script_dir = fileparts(mfilename('fullpath'));
out_png = fullfile(script_dir, 'bermudan_vs_european_q.png');
exportgraphics(gcf, out_png, 'Resolution', 300);