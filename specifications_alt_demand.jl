# This script describes the different specifications of dependence on observables and moment orderings that we are going to try

# - first, a helper function that builds the specification named tuple using the variables of interest:
function build_spec(spec)
    n97 = length(spec.vg)+1
    n02 = (length(spec.vg)+1)*2 + length(spec.vf) + length(spec.vm) + 2
    g_idx_97 = 1:n97
    zlist_97 =  [(spec.vg...,:logprice_c_m)]
    g_idx_02 = (n97+1):(n97+n02)
    zlist_02 = [(spec.vm...,:logprice_m_g),
    (spec.vf...,:logprice_f_g),
    (spec.vg...,:logprice_c_g), #<= here we are assuming that spec.vg ⊃ spec.vm and spec.vf
    (spec.vg...,:logprice_c_m)]
    return (vm = spec.vm, vf = spec.vf, vθ = spec.vθ, vg = spec.vg,
    g_idx_97 = g_idx_97, zlist_97 = zlist_97,
    g_idx_02 = g_idx_02, zlist_02 = zlist_02 
    )
end

# ------------ Dependence on Observables --------- #
# using just education
spec_1 = (vm = [:constant;:div;m_ed[2:3];f_ed[2:3];:age;:num_0_5],
        vf = [:constant;m_ed[2:3];f_ed[2:3];:age;:num_0_5],
        vθ = [:constant;:div;m_ed[2:3];f_ed[2:3];:age;:num_0_5],
        vg = [:constant;:div;m_ed[2:3];f_ed[2:3];:age;:num_0_5])

spec_1 = build_spec(spec_1)

# using just cluster dummies
spec_2 = build_spec((vm = [:constant;:div;cluster_dummies[2:end];f_ed[2:3];:age;:num_0_5],
        vf = [:constant;cluster_dummies[2:end];f_ed[2:3];:age;:num_0_5],
        vθ = [:constant,:div,:age,:num_0_5],
        vg = [:constant;:div;cluster_dummies[2:end];f_ed[2:3];:age;:num_0_5]))

# using cluster dummies and education
spec_3 = build_spec((vm = [:constant;:div;cluster_dummies[2:end];m_ed[2:3];f_ed[2:3];:age;:num_0_5],
        vf = [:constant;cluster_dummies[2:end];m_ed[2:3];f_ed[2:3];:age;:num_0_5],
        vθ = [:constant,:div,:age,:num_0_5],
        vg = [:constant;:div;cluster_dummies[2:end];m_ed[2:3];f_ed[2:3];:age;:num_0_5]))

# using the center estimates from a clustering exercise with more types
spec_4 = build_spec((vm = [:constant;:div;:mu_k;f_ed[2:3];:age;:num_0_5],
        vf = [:constant;:mu_k;f_ed[2:3];:age;:num_0_5],
        vθ = [:constant,:div,:age,:num_0_5],
        vg = [:constant;:div;:mu_k;f_ed[2:3];:age;:num_0_5]))

# using centers (as above) and education
spec_5 = build_spec((vm = [:constant;:div;:mu_k;m_ed[2:3];f_ed[2:3];:age;:num_0_5],
        vf = [:constant;:mu_k;m_ed[2:3];f_ed[2:3];:age;:num_0_5],
        vθ = [:constant,:div,:age,:num_0_5],
        vg = [:constant;:div;:mu_k;m_ed[2:3];f_ed[2:3];:age;:num_0_5]))

price_ratios = [:logprice_c_g;:logprice_m_g;:logprice_f_g]
interactions_1 = make_interactions(panel_data,price_ratios,spec_1.vm)
interactions_2 = make_interactions(panel_data,price_ratios,spec_2.vm)
interactions_3 = make_interactions(panel_data,price_ratios,spec_3.vm)
interactions_5 = make_interactions(panel_data,price_ratios,spec_5.vm)
        

