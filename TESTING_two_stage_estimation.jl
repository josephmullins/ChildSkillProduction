###########---------STEP 1: main-demand is run initially to get rel. demand parameters

include("main_demand.jl") #use this to get res2 
include("production.jl")

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
        ρ2 = x[1]
        γ2 = x[2]
        δ = x[3:4] #<- factor shares
        nm = length(spec.vm)
        βm = x[5:4+nm]
        pos = 5+nm
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
        P2 = CESmod(ρ=ρ2,γ=γ2,δ = δ,βm = βm,βf = βf,βg=βg,βθ=βθ,λ=λ,spec=spec)
        return P2
end

#keeping βm,βg,βf and ρ,γ from the first stage
#crude way to fix parameters without changing underlying functions?
function update_limited_p(y,x,spec)
        ρ = x[1]
        γ = x[2]
        δ = x[3:4] #<- factor shares
        nθ = length(spec.vθ)
        βθ = x[5:4+nθ]
        λ = x[end]
        P2 = CESmod(ρ=ρ,γ=γ,βm=y.βm,δ = δ,βf = y.βf,βg=y.βg,βθ=βθ,λ=λ,spec=spec)
        return P2
end


function update_inv(pars)
    @unpack ρ,γ,βm,βf,βg,βθ,δ,λ = pars
    return [ρ;γ;δ;βm;βf;βg;βθ;λ]
end


function initial_guess_p(spec)
        P = CESmod(spec)
        x0 = update_inv(P)
        x0[1:2] .= -2. #<- initial guess consistent with last time
        x0[3:4] = [0.1,0.9] #<- initial guess for δ
        return x0
end


###########---------STEP 3: testing a limited specification

spec_2p_2s = build_spec_prod_two_stage(
        (vm = spec_2.vm,vf = spec_2.vf, vg = spec_2.vg,vθ = spec_2.vm,
        zlist_prod_t = [0,5],
        zlist_prod = [[[interactions_2;:AP],[interactions_2;:LW],[:constant],[:constant]],[[:log_mtime],[:log_mtime],[],[]]])
)


##trying things while estimating a limited set of parameters 
p1=update(res2.est1,spec_2) #fix initial parameters from main_demand estimation
interactions_2s = make_interactions(panel_data,price_ratios,spec_2p_2s.vm) #panel_data updated for 
N = length(unique(panel_data.kid))
gfunc!(x,n,g,resids,data,spec) = production_moments_stacked!(p1,update_limited_p(p1,x,spec_2p_2s),n,g,resids,data,spec,true)
nmom = spec_2p_2s.g_idx_prod[end][end]
W = I(nmom)
x0 = [-2.;-2.;0.1;0.9;zeros(length(spec_2p_2s.vθ));1.]
gmm_criterion(x0,gfunc!,W,N,8,panel_data,spec_2p_2s)

res2s_limited,se2s_limited = estimate_gmm(x0,gfunc!,W,N,8,panel_data,spec_2p_2s)


###########---------Previous specifications 

p1=update(res2.est1,spec_2) #fix initial parameters from main_demand estimation
interactions_2s = make_interactions(panel_data,price_ratios,spec_2p_2s.vm) 

N = length(unique(panel_data.kid))

gfunc!(x,n,g,resids,data,spec) = production_moments_stacked!(p1,update_p(p1,x,spec_2p_2s),n,g,resids,data,spec,true)

nmom = spec_2p_2s.g_idx_prod[end][end]
W = I(nmom)
x0 = initial_guess_p(spec_2p_2s) 

gmm_criterion(x0,gfunc!,W,N,8,panel_data,spec_2p_2s)


res2s,se2s = estimate_gmm_iterative(x0,gfunc!,8,W,N,8,panel_data,spec_2p_2s)

res2s_gmm,se2s_gmm = estimate_gmm(x0,gfunc!,W,N,8,panel_data,spec_2p_2s) #this is not running any better





