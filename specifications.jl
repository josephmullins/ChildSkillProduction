# This script describes the different specifications of dependence on observables and moment orderings that we are going to try

# ------------ Dependence on Observables --------- #
# using just education
spec_1 = (vm = [:mar_stat;:div;m_ed[2:3];:age;:num_0_5],
        vf = [:const;f_ed[2:3];:age;:num_0_5],
        vθ = [:const,:mar_stat,:age,:num_0_5],
        vg = [:mar_stat;:div;m_ed[2:3];f_ed[2:3];:age;:num_0_5])

# using just cluster dummies
spec_2 = (vm = [:mar_stat;:div;cluster_dummies[2:nclusters];:age;:num_0_5],
        vf = [:const;f_ed[2:3];:age;:num_0_5],
        vθ = [:const,:mar_stat,:age,:num_0_5],
        vg = [:mar_stat;:div;cluster_dummies[2:nclusters];f_ed[2:3];:age;:num_0_5])

# using cluster dummies and education
spec_3 = (vm = [:mar_stat;:div;cluster_dummies[2:nclusters];m_ed[2:3];:age;:num_0_5],
        vf = [:const;f_ed[2:3];:age;:num_0_5],
        vθ = [:const,:mar_stat,:age,:num_0_5],
        vg = [:mar_stat;:div;cluster_dummies[2:nclusters];m_ed[2:3];f_ed[2:3];:age;:num_0_5])

# using the center estimates from a clustering exercise with more types
spec_4 = (vm = [:mar_stat;:div;:mu_k;:age;:num_0_5],
        vf = [:const;f_ed[2:3];:age;:num_0_5],
        vθ = [:const,:mar_stat,:age,:num_0_5],
        vg = [:mar_stat;:div;:mu_k;m_ed[2:3];f_ed[2:3];:age;:num_0_5])

# using centers (as above) and education
spec_5 = (vm = [:mar_stat;:div;:mu_k;m_ed[2:3];:age;:num_0_5],
        vf = [:const;f_ed[2:3];:age;:num_0_5],
        vθ = [:const,:mar_stat,:age,:num_0_5],
        vg = [:mar_stat;:div;:mu_k;m_ed[2:3];f_ed[2:3];:age;:num_0_5])


# -------------- Moment, Residual and Instrument Combinations ---------------------- #
# to read these functions, it helps to remember that the ordering of residuals is always [c/m,f/m,c/g,m/g,f/g] and the function calc_demand_resids will write a zero if data is missing for any reason

# ---- Version (1): this version stacks 97 and 02 moments on top of each other, and holds moments for married and divorced couples separately in the vector
# - doesn't use the ratio of father's to mother's time.
# - replicates what we did in first draft of the paper
function gmap_v1(data,it,spec)
    n97s = length(spec.vm)
    n02s = length(spec.vm)*3
    n97m = length(spec.vg)
    n02m = length(spec.vg)*2 + length(spec.vm) + length(spec.vf) + 1
    if !data.mar_stat[it]
        if data.year[it]==1997
            g_idx = 1:n97s
            zlist =  [[spec.vm[2:end];:logprice_c_m]]
            r_idx = [1]
        else
            g_idx = (1+n97s):(n97s+n02s)
            zlist = [[spec.vm[2:end];:logprice_m_g],
            [spec.vm[2:end];:logprice_c_g],
            [spec.vm[2:end];:logprice_c_m]]
            r_idx = [4,3,1]
        end
    else
        if data.year[it]==1997
            g_idx = (n97s+n02s+1):(n97s+n02s+n97m)
            zlist =  [[spec.vg[[1;3:end]];:logprice_c_m]]
            r_idx = [1] 
        else
            g_idx = (n97s+n02s+n97m+1):(n97s+n02s+n97m+n02m)
            zlist = [[spec.vm[[1;3:end]];:logprice_m_g],
            [spec.vf;:logprice_f_g],
            [spec.vg[[1;3:end]];:logprice_c_g],
            [spec.vg[[1;3:end]];:logprice_c_m]]
            r_idx = [4,5,3,1]
        end
    end
    return g_idx,zlist,r_idx
end

# ---- Version (2): this version puts married and single in the same moment instead of on top of each other
# tries to mimic the ordering above, but instead puts married and single in the same location of the g vector (when appropriate)
function gmap_v2(data,it,spec)
    n97 = length(spec.vg)+1
    n02 = (length(spec.vg)+1)*2 + length(spec.vf) + length(spec.vm) + 2
    if data.year[it]==1997
        g_idx = 1:n97
        zlist =  [[spec.vg;:logprice_c_m]]
        r_idx = [1] 
    else
        g_idx = (n97+1):(n97+n02)
        zlist = [[spec.vm;:logprice_m_g],
        [spec.vf;:logprice_f_g],
        [spec.vg;:logprice_c_g],
        [spec.vg;:logprice_c_m]]
        r_idx = [4,5,3,1]
    end
    return g_idx,zlist,r_idx
end
