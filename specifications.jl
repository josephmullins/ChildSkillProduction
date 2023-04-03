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
    (spec.vg...,:logprice_c_g),
    (spec.vg...,:logprice_c_m)]
    return (vm = spec.vm, vf = spec.vf, vθ = spec.vθ, vg = spec.vg,
    g_idx_97 = g_idx_97, zlist_97 = zlist_97,
    g_idx_02 = g_idx_02, zlist_02 = zlist_02 
    )
end

# ------------ Dependence on Observables --------- #
# using just education
spec_1 = (vm = [:mar_stat;:div;m_ed[2:3];:age;:num_0_5],
        vf = [:const;f_ed[2:3];:age;:num_0_5],
        vθ = [:const,:mar_stat,:age,:num_0_5],
        vg = [:mar_stat;:div;m_ed[2:3];f_ed[2:3];:age;:num_0_5])

spec_1 = build_spec(spec_1)

# using just cluster dummies
spec_2 = build_spec((vm = [:mar_stat;:div;cluster_dummies[2:nclusters];:age;:num_0_5],
        vf = [:const;f_ed[2:3];:age;:num_0_5],
        vθ = [:const,:mar_stat,:age,:num_0_5],
        vg = [:mar_stat;:div;cluster_dummies[2:nclusters];f_ed[2:3];:age;:num_0_5]))

# using cluster dummies and education
spec_3 = build_spec((vm = [:mar_stat;:div;cluster_dummies[2:nclusters];m_ed[2:3];:age;:num_0_5],
        vf = [:const;f_ed[2:3];:age;:num_0_5],
        vθ = [:const,:mar_stat,:age,:num_0_5],
        vg = [:mar_stat;:div;cluster_dummies[2:nclusters];m_ed[2:3];f_ed[2:3];:age;:num_0_5]))

# using the center estimates from a clustering exercise with more types
spec_4 = build_spec((vm = [:mar_stat;:div;:mu_k;:age;:num_0_5],
        vf = [:const;f_ed[2:3];:age;:num_0_5],
        vθ = [:const,:mar_stat,:age,:num_0_5],
        vg = [:mar_stat;:div;:mu_k;m_ed[2:3];f_ed[2:3];:age;:num_0_5]))

# using centers (as above) and education
spec_5 = build_spec((vm = [:mar_stat;:div;:mu_k;m_ed[2:3];:age;:num_0_5],
        vf = [:const;f_ed[2:3];:age;:num_0_5],
        vθ = [:const,:mar_stat,:age,:num_0_5],
        vg = [:mar_stat;:div;:mu_k;m_ed[2:3];f_ed[2:3];:age;:num_0_5]))



