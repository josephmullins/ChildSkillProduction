###########---------STEP 1: main-demand is run initially to get rel. demand parameters

x0 = initial_guess(spec_2)
nmom = spec_2.g_idx_02[end]
W = I(nmom)

res2 = estimate_gmm(x0,gfunc!,W,N,5,panel_data,spec_2) #initial relative demand parameters 

###########---------STEP 2: updated functions

##a function to build a specification that g_idx_prod begins at the first row of the moment vector
function build_spec_prod_two_stage(spec)
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
        g_idx_prod = g_idx_prod,zlist_prod_t = spec.zlist_prod_t,zlist_prod = zlist_prod
        )
    end
    
 
#this update function from estimate_production_demand_unrestricted, only returing P2
function update_p(x,spec)
        ρ = x[1]
        γ = x[2]
        ρ2 = x[3]
        γ2 = x[4]
        δ = x[5:6] #<- factor shares
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
        P1 = CESmod(ρ=ρ,γ=γ,βm = βm,βf = βf,βg=βg,spec=spec)
        P2 = CESmod(ρ=ρ2,γ=γ2,δ = δ,βm = βm,βf = βf,βg=βg,βθ=βθ,λ=λ,spec=spec)
        return P2
    end

    function initial_guess_p(spec)
        P = CESmod(spec)
        x0 = update_inv(P,P)
        x0[1:4] .= -2. #<- initial guess consistent with last time
        x0[5:6] = [0.1,0.9] #<- initial guess for δ
        return x0
    end


###########---------STEP 3: checking to see if things will run 

spec_2p_2s = build_spec_prod_two_stage(
        (vm = spec_2.vm,vf = spec_2.vf, vg = spec_2.vg,vθ = spec_2.vm,
        zlist_prod_t = [0,5],
        zlist_prod = [[[interactions_2;:AP],[interactions_2;:LW],[:constant],[:constant]],[[:log_mtime],[:log_mtime],[],[]]])
)

p1=update(res2.est1,spec_2) #fix initial parameters from main_demand estimation

N = length(unique(panel_data.kid))

gfunc!(x,n,g,resids,data,spec) = production_moments_stacked!(p1,update_p(x,spec_2p_2s),n,g,resids,data,spec) #this runs now!

nmom = spec_2p_2s.g_idx_prod[end][end]
W = I(nmom)
x0 = initial_guess_p(spec_2p_2s) 

gmm_criterion(x0,gfunc!,W,N,5,panel_data,spec_2p_2s)

res2s,se2s = estimate_gmm_iterative(x0,gfunc!,5,W,N,5,panel_data,spec_2p_2s)
