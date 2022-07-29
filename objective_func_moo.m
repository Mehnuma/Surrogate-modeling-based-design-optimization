function f = objective_func_moo(x)

global gpr_mass gpr_fatigue
f1 = predict(gpr_mass, x);
f2 = predict(gpr_fatigue, x);

f = [f1 1-f2];
end