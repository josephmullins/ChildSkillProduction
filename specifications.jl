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
    g_idx_07 = (n97+n02+1):(n97+2n02)
    return (vm = spec.vm, vf = spec.vf, vθ = spec.vθ, vg = spec.vg,
    g_idx_97 = g_idx_97, zlist_97 = zlist_97,
    g_idx_02 = g_idx_02, zlist_02 = zlist_02,
    g_idx_07 = g_idx_07 
    )
end

# ------------ Dependence on Observables --------- #
# using just education
spec_1 = (vm = [:constant;:div;m_ed[2:3];:age;:num_0_5],
        vf = [:constant;f_ed[2:3];:age;:num_0_5],
        vθ = [:constant;:div;m_ed[2:3];f_ed[2:3];:age;:num_0_5],
        vg = [:constant;:div;m_ed[2:3];f_ed[2:3];:age;:num_0_5])

spec_1 = build_spec(spec_1)

# using just cluster dummies
spec_2 = build_spec((vm = [:constant;:div;cluster_dummies[2:end];:age;:num_0_5],
        vf = [:constant;f_ed[2:3];:age;:num_0_5],
        vθ = [:constant,:div,:age,:num_0_5],
        vg = [:constant;:div;cluster_dummies[2:end];f_ed[2:3];:age;:num_0_5]))

# using cluster dummies and education
spec_3 = build_spec((vm = [:constant;:div;cluster_dummies[2:end];m_ed[2:3];:age;:num_0_5],
        vf = [:constant;f_ed[2:3];:age;:num_0_5],
        vθ = [:constant,:div,:age,:num_0_5],
        vg = [:constant;:div;cluster_dummies[2:end];m_ed[2:3];f_ed[2:3];:age;:num_0_5]))

# using the center estimates from a clustering exercise with more types
spec_4 = build_spec((vm = [:constant;:div;:mu_k;:age;:num_0_5],
        vf = [:constant;f_ed[2:3];:age;:num_0_5],
        vθ = [:constant,:div,:age,:num_0_5],
        vg = [:constant;:div;:mu_k;f_ed[2:3];:age;:num_0_5]))

# using centers (as above) and education
spec_5 = build_spec((vm = [:constant;:div;:mu_k;m_ed[2:3];:age;:num_0_5],
        vf = [:constant;f_ed[2:3];:age;:num_0_5],
        vθ = [:constant,:div,:age,:num_0_5],
        vg = [:constant;:div;:mu_k;m_ed[2:3];f_ed[2:3];:age;:num_0_5]))


# a second helper function that builds a specification as above, but does so including info relevant to production
function build_spec_prod(spec)
        n97 = length(spec.vg)+1
        n02 = (length(spec.vg)+1)*2 + length(spec.vf) + length(spec.vm) + 2
        g_idx_97 = 1:n97
        zlist_97 =  [(spec.vg...,:logprice_c_m)]
        g_idx_02 = (n97+1):(n97+n02)
        zlist_02 = [(spec.vm...,:logprice_m_g),
        (spec.vf...,:logprice_f_g),
        (spec.vg...,:logprice_c_g), #<= here we are assuming that spec.vg ⊃ spec.vm and spec.vf
        (spec.vg...,:logprice_c_m)]
        g_idx_07 = (n97+n02+1):(n97+2n02)

        # create the positions in which to write moments for each t
        g_idx_prod_97 = []
        gpos = (n97+2n02)
        for t in eachindex(spec.zlist_prod)
                
                K = sum(length(z) for z in spec.zlist_prod[t]) #<- number of moments
                push!(g_idx_prod_97,gpos+1:gpos+K)
                gpos += K
        end

        g_idx_prod_02 = []
        for t in eachindex(spec.zlist_prod)                
                K = sum(length(z) for z in spec.zlist_prod[t]) #<- number of moments
                push!(g_idx_prod_02,gpos+1:gpos+K)
                gpos += K
        end


        return (vm = spec.vm, vf = spec.vf, vθ = spec.vθ, vg = spec.vg,
        g_idx_97 = g_idx_97, zlist_97 = zlist_97,
        g_idx_02 = g_idx_02, zlist_02 = zlist_02,
        g_idx_07 = g_idx_07,
        g_idx_prod_97 = g_idx_prod_97, g_idx_prod_02 = g_idx_prod_02,
        zlist_prod_t = spec.zlist_prod_t,zlist_prod = spec.zlist_prod
        )
end


price_ratios = [:logprice_c_g;:logprice_m_g;:logprice_f_g]
interactions_1 = make_interactions(panel_data,price_ratios,spec_1.vg)
input_instruments = [:log_mtime,:log_ftime_coalesced,:log_chcare_input,:log_good_input] # 

spec_1p_x = build_spec_prod(
        (vm = spec_1.vm,vf = spec_1.vf, vg = spec_1.vg,vθ = spec_1.vg,
        zlist_prod_t = [0,5],
        zlist_prod = [[[spec_1.vg;:log_total_income;interactions_1;:LW],[spec_1.vg;:log_total_income;interactions_1;:AP],[],[],[],[],[:constant],[:constant]],[[:log_mtime],[:log_mtime],[],[],[],[],[],[]]])
)

interactions_2 = make_interactions(panel_data,price_ratios,spec_2.vg)

spec_2p_x = build_spec_prod(
        (vm = spec_2.vm,vf = spec_2.vf, vg = spec_2.vg,vθ = spec_2.vg,
        zlist_prod_t = [0,5],
        zlist_prod = [[[spec_2.vg;:log_total_income;interactions_2;:LW],[spec_2.vg;:log_total_income;interactions_2;:AP],[],[],[],[],[:constant],[:constant]],[[:log_mtime],[:log_mtime],[],[],[],[],[],[]]])
)

interactions_3 = make_interactions(panel_data,price_ratios,spec_3.vg)

spec_3p_x = build_spec_prod((vm = spec_3.vm,vf = spec_3.vf, vg = spec_3.vg,vθ = spec_3.vg,
zlist_prod_t = [0,5],
zlist_prod = [[[spec_3.vg;:log_total_income;interactions_3;:LW],[spec_3.vg;:log_total_income;interactions_3;:AP],[],[],[],[],[:constant],[:constant]],[[:log_mtime],[:log_mtime],[],[],[],[],[],[]]]))

interactions_5 = make_interactions(panel_data,price_ratios,spec_5.vg)

spec_5p_x = build_spec_prod(
        (vm = spec_5.vm,vf = spec_5.vf, vg = spec_5.vg,vθ = spec_5.vg,
        zlist_prod_t = [0,5],
        zlist_prod = [[[spec_5.vg;:log_total_income;interactions_5;:LW],[spec_5.vg;:log_total_income;interactions_5;:AP],[],[],[],[],[:constant],[:constant]],[[:log_mtime],[:log_mtime],[],[],[],[],[],[]]])
)


spec_1p=spec_1p_x
spec_2p=spec_2p_x