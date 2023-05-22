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
spec_1 = (vm = [:constant;:div;m_ed[2:3];:age;:num_0_5],
        vf = [:constant;f_ed[2:3];:age;:num_0_5],
        vθ = [:constant,:div,:age,:num_0_5],
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
        # at vθ to zlist for production moments 
        zlist_prod = spec.zlist_prod
        for v in reverse(spec.vg) #<- assuming vg ⊃ vθ,vf,vm. Call union function instead? 
                pushfirst!(zlist_prod[1][1],v)
                pushfirst!(zlist_prod[1][2],v)
        end                

        # create the positions in which to write moments for each t
        g_idx_prod = []
        
        gpos = (n97+n02)
        for t in eachindex(spec.zlist_prod)
                K = sum(length(z) for z in spec.zlist_prod[t]) #<- number of moments
                push!(g_idx_prod,gpos+1:gpos+K)
                gpos += K
        end
        
        return (vm = spec.vm, vf = spec.vf, vθ = spec.vθ, vg = spec.vg,
        g_idx_97 = g_idx_97, zlist_97 = zlist_97,
        g_idx_02 = g_idx_02, zlist_02 = zlist_02,
        g_idx_prod = g_idx_prod,zlist_prod_t = spec.zlist_prod_t,zlist_prod = zlist_prod
        )
end
# NEXT: test the moment functions with all of this

# using mother's education and using just prices in 97 as production instruments:

price_ratios = [:logprice_c_g;:logprice_m_g;:logprice_f_g]
interactions_1 = make_interactions(panel_data,price_ratios,spec_1.vm)

spec_1p_x = build_spec_prod(
        (vm = spec_1.vm,vf = spec_1.vf, vg = spec_1.vg,vθ = spec_1.vm,
        zlist_prod_t = [0,5],
        zlist_prod = [[[interactions_1;:AP],[interactions_1;:LW],[:constant],[:constant]],[[:log_mtime],[:log_mtime],[],[]]])
)


interactions_2 = make_interactions(panel_data,price_ratios,spec_2.vm)

spec_2p_x = build_spec_prod(
        (vm = spec_2.vm,vf = spec_2.vf, vg = spec_2.vg,vθ = spec_2.vm,
        zlist_prod_t = [0,5],
        zlist_prod = [[[interactions_2;:AP],[interactions_2;:LW],[:constant],[:constant]],[[:log_mtime],[:log_mtime],[],[]]])
)

interactions_3 = make_interactions(panel_data,price_ratios,spec_3.vm)

spec_3p_x = build_spec_prod(
        (vm = spec_3.vm,vf = spec_3.vf, vg = spec_3.vg,vθ = spec_3.vm,
        zlist_prod_t = [0,5],
        zlist_prod = [[[interactions_3;:AP],[interactions_3;:LW],[:constant],[:constant]],[[:log_mtime],[:log_mtime],[],[]]])
)


spec_1p=spec_1p_x
spec_2p=spec_2p_x
spec_3p=spec_3p_x