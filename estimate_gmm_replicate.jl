include("estimation_tools.jl")
include("relative_demand.jl")

# read in the data:
# Step 1: create the data object
D2 = DataFrame(CSV.File("CLMP_v1/data/gmm_full_vertical.csv",missingstring = "NA"))
#d = DataFrame(CSV.File("data/gmm_full_horizontal.csv",missingstring = "NA"))
D2[!,:mar_stat] = D2.mar_stable
D2[!,:logwage_m] = log.(D2.m_wage)
#D2[!,:age_sq] = D2.age_mother.^2
D2[!,:logwage_f] = log.(D2.f_wage)
D2[!,:logprice_g] = log.(D2.price_g)
D2[!,:logprice_c] = log.(D2.p_4f) # alternative is p_4c
#D2[!,:log_chcare] = replace(log.(D2.chcare),-Inf => missing)
D2[!,:log_mtime] = D2.log_mtime .- D2.logwage_m
D2[!,:log_ftime] = D2.log_ftime .- D2.logwage_f
#D2 = D2[.!ismissing.(D2.mar_stat),:]
D2 = D2[.!ismissing.(D2.price_g),:]
D2 = D2[.!ismissing.(D2.m_wage),:]
D2 = D2[.!(D2.mar_stat .& ismissing.(D2.f_wage)),:]
# D2[!,:goods] = D2.Toys .+ D2.tuition .+ D2.comm_grps .+ D2.lessons .+ D2.tutoring .+ D2.sports .+ D2.SchSupplies
# D2[!,:log_good] = replace(log.(D2.goods),-Inf => missing)
D2.m_ed = replace(D2.m_ed,">16" => "16","<12" => "12")
D2.f_ed = replace(D2.f_ed,">16" => "16","<12" => "12")
m_ed = make_dummy(D2,:m_ed)
f_ed = make_dummy(D2,:f_ed)
D2[.!D2.mar_stat,f_ed] .= 0. #<- make zero by default. this won't work all the time
D2 = D2[.!ismissing.(D2.age),:]
D2.logwage_f = coalesce.(D2.logwage_f,0) #<- make these into zeros to avoid a problem with instruments


#---- write the update function:
function update(x,spec)
    ρ = x[1]
    γ = x[2]
    nm = length(spec.vm)
    βm = x[3:2+nm]
    pos = 3+nm
    nf = length(spec.vf)
    βf = x[pos:pos+nf-1]
    ng = length(spec.vg)
    pos += nf
    βg = x[pos:pos+ng-1]
    return CESmod(ρ=ρ,γ=γ,βm = βm,βf = βf,βg=βg,spec=spec)
end
function update_inv(pars)
    @unpack ρ,γ,βm,βf,βg = pars
    return [ρ;γ;βm;βf;βg]
end

# ---- write the specification we want to use:
spec = (vm = [:mar_stat;:age;:num_0_5;m_ed],vf = [:age;:num_0_5;f_ed],vθ = [:const,:mar_stat,:age,:num_0_5],vg = [:mar_stat;:age;:num_0_5;m_ed;f_ed[2:end]]) 
P = CESmod(spec)
x0 = update_inv(P)
# ordering of residuals: c/m,f/m,c/g,m/g,f/g
# -- make some price ratios to pass as instruments:
D2[!,:logprice_c_m] = D2.logprice_c .- D2.logwage_m
D2[!,:logprice_m_f] = D2.logwage_m .- D2.logwage_f
D2[!,:logprice_c_g] = D2.logprice_c .- D2.logprice_g
D2[!,:logprice_m_g] = D2.logwage_m .- D2.logprice_g
D2[!,:logprice_f_g] = D2.logwage_f .- D2.logprice_g


# married: c/m,c/g,m/g,f/g
# single: c/m,c/g,m/g
Zm_list = [[spec.vm[2:end];:logprice_c_m],
    [spec.vg[2:end];:logprice_c_g],
    [spec.vg[2:end];:logprice_m_g],
    [spec.vg[2:end];:logprice_f_g]]

Zs_list = [[spec.vm[2:end];:logprice_c_m],
    [spec.vm[2:end];:logprice_c_g],
    [spec.vm[2:end];:logprice_m_g]]

# ---- write the moment function:
D2 = subset(D2,:year => x->(x.==1997) .| (x.==2002))

# we'll write a better version if we need to later
function custom_gmm(pars,data,Zm_list,Zs_list,W,N) 
    gd = groupby(data,:KID)
    #nmom = size(W)[1]
    #nmom_married = sum([length(z) for z in Zm_list]) + length(Zm_list[1])
    g97s = zeros(length(Zs_list[1]))
    g97m = zeros(length(Zm_list[1]))
    g02s = zeros(sum([length(z) for z in Zs_list]))
    g02m = zeros(sum([length(z) for z in Zm_list]))
    R = zeros(4)
    
    for i=1:gd.ngroups
        for it=gd.starts[i]:gd.ends[i]
            R[:] .= 0.
            lϕm,lϕf,lϕc,log_price_index,Φg = log_input_ratios(pars,data,it) #<
            if D.mar_stat[it]
                if !ismissing(data.log_mtime[it]) & !ismissing(data.logwage_m[it])
                    if !ismissing(data.log_chcare[it])
                        R[1] = data.log_chcare[it] - data.log_mtime[it] - (lϕc - lϕm) - data.logprice_c[it]
                    end
                end
                if !ismissing(data.log_good[it])
                    if !ismissing(data.log_chcare[it])
                        R[2] = data.log_chcare[it] - data.log_good[it] - lϕc - (data.logprice_c[it] - data.logprice_g[it])
                    end
                    if !ismissing(data.log_mtime[it]) #& !ismissing(data.logwage_m[it]) #<- include for missing wage?
                        R[3] = data.log_mtime[it] - data.log_good[it] - lϕm + data.logprice_g[it]
                    end
                    if !ismissing(data.log_ftime[it]) #& !ismissing(data.logwage_f[it]) #<- include for missing wage?
                        R[4] = data.log_ftime[it] - data.log_good[it] - lϕf + data.logprice_g[it]
                    end
                end
                if D.year[it]==1997
                    stack_moments!(g97m,R,data,[Zm_list[1]],it)
                else
                    stack_moments!(g02m,R,data,Zm_list,it)
                end
            else
                if !ismissing(data.log_mtime[it]) & !ismissing(data.logwage_m[it])
                    if !ismissing(data.log_chcare[it])
                        R[1] = data.log_chcare[it] - data.log_mtime[it] - (lϕc - lϕm) - data.logprice_c[it]
                    end
                end
                if !ismissing(data.log_good[it])
                    if !ismissing(data.log_chcare[it])
                        R[2] = data.log_chcare[it] - data.log_good[it] - lϕc - (data.logprice_c[it] - data.logprice_g[it])
                    end
                    if !ismissing(data.log_mtime[it]) #& !ismissing(data.logwage_m[it]) #<- include for missing wage?
                        R[3] = data.log_mtime[it] - data.log_good[it] - lϕm + data.logprice_g[it]
                    end
                end
                if D.year[it]==1997
                    stack_moments!(g97s,R,data,[Zs_list[1]],it)
                else
                    stack_moments!(g02s,R,data,Zs_list,it)
                end
            end
        end
    end
    g = [g97m;g02m;g97s;g02s] / N #<- take sample mean
    return g'*W*g
end
nmom = sum([length(z) for z in Zm_list])+sum([length(z) for z in Zs_list]) + length(Zm_list[1]) + length(Zs_list[1])
N = length(unique(D2.KID))
W = I(nmom)
x0[1:2] .= -2.
custom_gmm(update(x0,spec),D2,Zm_list,Zs_list,W,N) 

break

gstore = zeros(nmom)
rvec = zeros(Real,5) #<- doesn't seem to cost anything time wise
# PROBLEM: we need the type of rvec to change, but we also need the storage to be re-set each time

gfunc!(x,g,data,n) = stack_gmm!(x,g,rvec,resid!,data,Z_list,n)
W = I(nmom)
N = nrow(D2)

# res = optimize(x->gmm_criterion(x,gfunc!,D2,W,N),x0,Newton(),autodiff=:forward,Optim.Options(show_trace=true))

# @time gmm_criterion(x0,gfunc!,D2,W,N)
x0[1:2] = [-2.,-2.]
res,se = estimate_gmm_iterative(x0,gfunc!,D2,W,N,6)

# V = parameter_variance_gmm(res.minimizer,gfunc!,D2,W,N)
# dg = ForwardDiff.jacobian(x->moment_func(x,gfunc!,D2,N,nmom),res.minimizer)
# Σ = moment_variance(res.minimizer,gfunc!,D2,N,nmom)
# ALSO: check initial guess to see if we can replicate

# NEXT: figure out how to drop  m-f moment
# NEXT: figure out how to stack 97 and 02 moments (see if it makes a difference to estimates)
