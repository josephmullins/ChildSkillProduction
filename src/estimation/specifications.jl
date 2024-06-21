# This script describes the different specifications of dependence on observables and moment orderings that we are going to try

# - first, a helper function that builds the specification named tuple using the variables of interest:
# order of residuals for demand is:
# c / m, c / g, m / g, f / g
function build_spec(spec)
    n97 = length(spec.vy)+1
    n02 = (length(spec.vy)+1)*2 + length(spec.vf) + length(spec.vm) + 2
    zlist_97 =  [(spec.vy...,:logprice_c_m)]
    zlist_02 = [
    (spec.vy...,:logprice_c_m),
    (spec.vy...,:logprice_c_g), #<= here we are assuming that spec.vy ⊃ spec.vm and spec.vf
    (spec.vm...,:logprice_m_g),
    (spec.vf...,:logprice_f_g)
    ]
    zlist_07 = [
    (:constant,:logprice_c_m),
    (:constant,:logprice_c_g), 
    (:constant,:logprice_m_g),
    (:constant,:logprice_f_g)        
    ]
    zlist_prod_t = [0,5]
    zlist_prod = [[[spec.vy;:AP],[:log_mtime]],[[spec.vy;:AP],[:log_mtime]],[],[],[],[],[[:constant],[]],[[:constant],[]]]

    return (;vm = spec.vm, vf = spec.vf, vθ = [spec.vy;:ind02], vy = spec.vy,
    zlist_97, zlist_02, zlist_07, zlist_prod, zlist_prod_t
    )
end

# this functions outputs the four specifications we settled on.
function get_specifications(m_ed,f_ed,cluster_dummies)
    # using just education
    spec_1 = build_spec((vm = [:constant;:div;m_ed[2:3];:age;:num_0_5],
        vf = [:constant;f_ed[2:3];:age;:num_0_5],
        vθ = [:constant;:div;m_ed[2:3];f_ed[2:3];:age;:num_0_5],
        vy = [:constant;:div;m_ed[2:3];f_ed[2:3];:age;:num_0_5]))

    # using just cluster dummies
    spec_2 = build_spec((vm = [:constant;:div;cluster_dummies[2:end];:age;:num_0_5],
        vf = [:constant;f_ed[2:3];:age;:num_0_5],
        vθ = [:constant,:div,:age,:num_0_5],
        vy = [:constant;:div;cluster_dummies[2:end];f_ed[2:3];:age;:num_0_5]))

    # using cluster dummies and education
    spec_3 = build_spec((vm = [:constant;:div;cluster_dummies[2:end];m_ed[2:3];:age;:num_0_5],
        vf = [:constant;f_ed[2:3];:age;:num_0_5],
        vθ = [:constant,:div,:age,:num_0_5],
        vy = [:constant;:div;cluster_dummies[2:end];m_ed[2:3];f_ed[2:3];:age;:num_0_5]))

    # using centers (as above) and education
    spec_4 = build_spec((vm = [:constant;:div;:mu_k;m_ed[2:3];:age;:num_0_5],
        vf = [:constant;f_ed[2:3];:age;:num_0_5],
        vθ = [:constant,:div,:age,:num_0_5],
        vy = [:constant;:div;:mu_k;m_ed[2:3];f_ed[2:3];:age;:num_0_5]))
    return spec_1,spec_2,spec_3,spec_4
end
