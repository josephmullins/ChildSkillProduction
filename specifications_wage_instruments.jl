# This script describes the different specifications of dependence on observables and moment orderings that we are going to try

# - first, a helper function that builds the specification named tuple using the variables of interest:
# order of residuals for demand is:
# c / m, c / g, m / g, f / g
# m_pred_lnwage_mean_occ_state or f_pred_lnwage_mean_occ_state
function build_spec(spec)
        n97 = length(spec.vy)+1
        n02 = (length(spec.vy)+1)*2 + length(spec.vf) + length(spec.vm) + 2
        zlist_97 =  [(spec.vy...,:logprice_c,:m_pred_lnwage_mean_occ_state)]
        zlist_02 = [
        (spec.vy...,:logprice_c,:m_pred_lnwage_mean_occ_state),
        (spec.vy...,:logprice_c_g), #<= here we are assuming that spec.vy ⊃ spec.vm and spec.vf
        (spec.vm...,:m_pred_lnwage_mean_occ_state),
        (spec.vf...,:f_pred_lnwage_mean_occ_state)
        ]
        zlist_07 = [
                (:constant,:logprice_c,:m_pred_lnwage_mean_occ_state),
                (:constant,:logprice_c_g), 
                (:constant,:m_pred_lnwage_mean_occ_state),
                (:constant,:f_pred_lnwage_mean_occ_state)        
                ]        
        return (vm = spec.vm, vf = spec.vf, vθ = spec.vθ, vy = spec.vy,
        zlist_97 = zlist_97,
        zlist_02 = zlist_02,
        zlist_07 = zlist_07,
        zlist_prod = [],zlist_prod_t = []
        )
    end

# ------------ Dependence on Observables --------- #
# using just education
spec_1 = (vm = [:constant;:div;m_ed[2:3];:age;:num_0_5],
        vf = [:constant;f_ed[2:3];:age;:num_0_5],
        vθ = [:constant;:div;m_ed[2:3];f_ed[2:3];:age;:num_0_5],
        vy = [:constant;:div;m_ed[2:3];f_ed[2:3];:age;:num_0_5])

spec_1 = build_spec(spec_1)

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

# using the center estimates from a clustering exercise with more types
spec_4 = build_spec((vm = [:constant;:div;:mu_k;:age;:num_0_5],
        vf = [:constant;f_ed[2:3];:age;:num_0_5],
        vθ = [:constant,:div,:age,:num_0_5],
        vy = [:constant;:div;:mu_k;f_ed[2:3];:age;:num_0_5]))

# using centers (as above) and education
spec_5 = build_spec((vm = [:constant;:div;:mu_k;m_ed[2:3];:age;:num_0_5],
        vf = [:constant;f_ed[2:3];:age;:num_0_5],
        vθ = [:constant,:div,:age,:num_0_5],
        vy = [:constant;:div;:mu_k;m_ed[2:3];f_ed[2:3];:age;:num_0_5]))


price_ratios = [:logprice_c_g;:logprice_m_g;:logprice_f_g]
interactions_1 = make_interactions(panel_data,price_ratios,spec_1.vy)
input_instruments = [:log_mtime,:log_ftime_coalesced,:log_chcare_input,:log_good_input] # 
prices = [:m_pred_lnwage_mean_occ_state,:f_pred_lnwage_mean_occ_state]

spec_1p_x = 
        (vm = spec_1.vm,vf = spec_1.vf, vy = spec_1.vy,vθ = [spec_1.vy;:ind02],
        zlist_97 = spec_5.zlist_97,
        zlist_02 = spec_5.zlist_02,
        zlist_07 = spec_5.zlist_07,    
        zlist_prod_t = [0,5],
        zlist_prod = [[[spec_1.vy;prices;:AP],[:log_mtime]],[[spec_1.vy;prices;:AP],[:log_mtime]],[],[],[],[],[[:constant],[]],[[:constant],[]]])

interactions_2 = make_interactions(panel_data,price_ratios,spec_2.vy)

spec_2p_x = (
    vm = spec_2.vm,vf = spec_2.vf, vy = spec_2.vy,vθ = [spec_2.vy;:ind02],
    zlist_97 = spec_5.zlist_97,
    zlist_02 = spec_5.zlist_02,
    zlist_07 = spec_5.zlist_07,
    zlist_prod_t = [0,5],
    zlist_prod = [[[spec_2.vy;prices;:AP],[:log_mtime]],[[spec_2.vy;prices;:AP],[:log_mtime]],[],[],[],[],[[:constant],[]],[[:constant],[]]])

interactions_3 = make_interactions(panel_data,price_ratios,spec_3.vy)

spec_3p_x = (
    vm = spec_3.vm,vf = spec_3.vf, vy = spec_3.vy,vθ = [spec_3.vy;:ind02],
    zlist_97 = spec_5.zlist_97,
    zlist_02 = spec_5.zlist_02,
    zlist_07 = spec_5.zlist_07,
    zlist_prod_t = [0,5],
    zlist_prod = [[[spec_3.vy;prices;:AP],[:log_mtime]],[[spec_3.vy;prices;:AP],[:log_mtime]],[],[],[],[],[[:constant],[]],[[:constant],[]]])


interactions_5 = make_interactions(panel_data,price_ratios,spec_5.vy)

spec_5p_x = (
    vm = spec_5.vm,vf = spec_5.vf, vy = spec_5.vy,vθ = [spec_5.vy;:ind02],
    zlist_97 = spec_5.zlist_97,
    zlist_02 = spec_5.zlist_02,
    zlist_07 = spec_5.zlist_07,
    zlist_prod_t = [0,5],
    zlist_prod = [[[spec_5.vy;prices;:AP],[:log_mtime]],[[spec_5.vy;prices;:AP],[:log_mtime]],[],[],[],[],[[:constant],[]],[[:constant],[]]])

