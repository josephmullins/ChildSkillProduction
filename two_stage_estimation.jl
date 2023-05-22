
###########---------STEP 1: estimate the relative demand parameters as in main demand, using spec2 here
#currently, I was just testing things with spec_2 from main demand

x0 = initial_guess(spec_2)
nmom = spec_2.g_idx_02[end]
W = I(nmom)

res2 = estimate_gmm(x0,gfunc!,W,N,5,panel_data,spec_2) #initial relative demand parameters 


###########---------STEP 2: setting the values for pars1 and pars2 using build_spec_prod_two_stage

y=res2.est1 #y is used to fix relative demand parameters 
x0=initial_guess(spec_2p_x_2s)

p1=update(res2.est1,spec_2) #pars1
p2=update_two_stage(y,x0,spec_2p_x_2s) #pars2

#this update function fixes the relative demand parameters
function update_two_stage(y,x,spec)
    ρ = y[1]
    γ = y[2]
    ρ2 = y[3] 
    γ2 = y[4]
    δ = y[5:6] #<- factor shares
    nm = length(spec.vm)
    βm = x[7:6+nm]
    pos = 7+nm
    nf = length(spec.vf)
    βf = x[pos:pos+nf-1]
    ng = length(spec.vg)
    pos += nf
    βg = x[pos:pos+ng-1]
    pos += ng
    nθ = length(spec.vθ)
    βθ = x[pos:pos+nθ-1]
    pos+= nθ
    λ = x[pos]
    P1 = CESmod(ρ=ρ,γ=γ,δ = δ,βm = βm,βf = βf,βg=βg,spec=spec)
    P2 = CESmod(ρ=ρ,γ=γ,δ = δ,βm = βm,βf = βf,βg=βg,βθ=βθ,λ=λ,spec=spec)
    return P2
end

###########---------STEP 3: checking to see if things will run 

N = length(unique(panel_data.kid))

gfunc!(x,n,g,resids,data,spec) = production_demand_moments_stacked!(p1,p2,n,g,resids,data,spec)

nmom = spec_2p_x.g_idx_prod[end][end]
W = I(nmom)
x0 = initial_guess(spec_2p_x_2s)

###spec_2p is equivalent to spec2p_x
##5 is nresids
gmm_criterion(x0,gfunc!,W,N,5,panel_data,spec_2p_x_2s)

res2s,se2s = estimate_gmm_iterative(x0,gfunc!,5,W,N,5,panel_data,spec_2p_x)
#still getting some crazy standard errors for this 


###########---------FUNCTIONS/SPECIFICATION: 

##a function to build a specification that g_idx_prod begins at the first row of the moment vector
function build_spec_prod_two_stage(spec)
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
    
    gpos = 0
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

spec_2p_x_2s = build_spec_prod_two_stage(
        (vm = spec_2.vm,vf = spec_2.vf, vg = spec_2.vg,vθ = spec_2.vm,
        zlist_prod_t = [0,5],
        zlist_prod = [[[interactions_2;:AP],[interactions_2;:LW],[:constant],[:constant]],[[:log_mtime],[:log_mtime],[],[]]])
)

