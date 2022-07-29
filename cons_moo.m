
% Constraint function for the multi-objective optimization

function [c, ceq] = cons_moo(x)

global gpr_np_fd gpr_np_vg gpr_mass 
np_fd = predict(gpr_np_fd, x);
np_vg = predict(gpr_np_vg, x);

f1 = predict(gpr_mass, x);

% Calculate Wear (Constant values have been changed)
k = 4e-4; 
s = 5; % mm
h = 150*9.8; % N/mm^2 Brinnell hardness 

W_fd = k*np_fd.*s/h;
W_vg = k*np_vg.*s/h; 
W_fd_max = 0.5;
W_vg_max = 0.2;

c(1) = W_fd -W_fd_max;
c(2) = W_vg -W_vg_max;
c(3) = -f1 +0.0035;
c(4) = f1 -1;

ceq = [];
end