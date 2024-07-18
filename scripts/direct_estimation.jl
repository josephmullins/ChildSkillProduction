include("../src/model.jl")
include("../src/model_older_children.jl")
include("../src/estimation.jl")
include("../src/estimation/direct_method_1.jl")

# =======================   read in the data and estimates ===================================== #
# - load the data 
panel_data = DataFrame(CSV.File("../../../PSID_CDS/data-derived/psid_fam.csv",missingstring = ["","NA"]))
panel_data, m_ed, f_ed = prep_data(panel_data)
panel_data = DataFrame(filter(x-> sum(skipmissing(x.ind_not_sample.==0))>0 || sum(x.all_prices)>0,groupby(panel_data,:kid)))
wage_types = DataFrame(CSV.File("data/wage_types.csv"))

panel_data=innerjoin(panel_data, wage_types, on = :MID) #merging in cluster types
cluster_dummies=make_dummy(panel_data,:cluster) #cluster dummies made

# ======= Introduce alternative normalization of test scores ====== #
scores = DataFrame(CSV.File("../../../PSID_CDS/data-cds/assessments/AssessmentPanel.csv",missingstring=["","NA"]))
scores = select(scores,[:KID,:year,:LW_raw,:AP_raw,:AP_std,:LW_std])
scores = rename(scores,:KID => :kid)
panel_data = sort(leftjoin(panel_data,scores,on=[:kid,:year]),[:kid,:year])
mLW = mean(skipmissing(panel_data.LW_raw[panel_data.age.==12]))
sLW = std(skipmissing(panel_data.LW_raw[panel_data.age.==12]))
mAP = mean(skipmissing(panel_data.AP_raw[panel_data.age.==12]))
sAP = std(skipmissing(panel_data.AP_raw[panel_data.age.==12]))

using DataFramesMeta
panel_data = @chain panel_data begin
    # groupby(:age)
    @transform :LW = (:LW_raw .- mLW)/sLW :AP = (:AP_raw .- mAP)/sAP
end


# ============= Write the specification =================== #
vm = [:constant] #[:constant;:div;m_ed[2:3];:age;:num_0_5]
vf = [:constant] #;f_ed[2:3];:age;:num_0_5]
vθ = [:constant;:div;m_ed[2:3];f_ed[2:3];:age;:num_0_5;:ind02]
vy = [:constant;:div] #;m_ed[2:3];f_ed[2:3];:age;:num_0_5]
zc = [:constant;:div;m_ed[2:3];f_ed[2:3];:age;:num_0_5]
zf = [:constant;m_ed[2:3];f_ed[2:3];:age;:num_0_5]
zg = [:constant;:div;m_ed[2:3];f_ed[2:3];:age;:num_0_5]
zτ = [:age,:num_0_5]
vΩ = [:logprice_c_g,:logprice_f_g,:logprice_m_g,:logprice_g,:log_total_income]
zlist_97 = [(zc...,vΩ...),(zf...,vΩ...),(zτ...,vΩ...)]
zlist_02 = [(zc...,vΩ...),(zg...,vΩ...),(zf...,vΩ...),(zτ...,vΩ...)]
zlist_07 = [(zc...,vΩ...),(zf...,vΩ...)]
zlist_prod_t = [0,5]
zlist_prod = [[[vy;vΩ;:AP],[:log_mtime]],[[vy;vΩ;:AP],[:log_mtime]],[[:constant],[]],[[:constant],[]]]
spec = (;vm,vf,vθ,vy,zc,zf,zg,zτ,vΩ,zlist_97,zlist_02,zlist_07,zlist_prod_t,zlist_prod)

# ======= Step 1: Estimate the Demand Parameters ======== #
spec = (;vm,vf,vθ,vy,zc,zf,zg,zτ,vΩ,zlist_97,zlist_02,zlist_07,zlist_prod_t = [],zlist_prod = [])
data = child_data(panel_data,spec)
gfunc!(x,n,g,resids,data,spec) = demand_moments_method_1!(update_demand_method_1(x,spec),n,g,resids,data)
x0 = zeros(sum(length(x) for x in [spec.zc,spec.zf,spec.zg,spec.zτ])+20)

nmom = sum([size(z,1)*!isempty(z) for z in data.Z])
nresids = length(data.Z)
N = length(unique(panel_data.kid))
W = I(nmom)
res = estimate_gmm(x0,gfunc!,W,N,nresids,data,spec)
dp = update_demand_method_1(res.est,spec)

# ======= Step 2: Run a bootstrap with identity weighting matrix ========= #
spec = (;vm,vf,vθ,vy,zc,zf,zg,zτ,vΩ,zlist_97=[],zlist_02=[],zlist_07=[],zlist_prod_t,zlist_prod)
data = child_data(panel_data,spec)
gfunc!(x,n,g,resids,data,spec) = production_moments_method_1!(dp,update(x,spec,"uc"),n,g,resids,data)
nmom = sum([size(z,1)*!isempty(z) for z in data.Z])
nresids = length(data.Z)
N = length(unique(panel_data.kid))
W = I(nmom)
ntrials = 50
xp = initial_guess(spec,"uc");
Xb = zeros(length(xp),ntrials)
Random.seed!(71123)
for b in 1:ntrials
    ib = rand(1:N,N)
    @show b
    res = optimize(x->gmm_criterion(x,gfunc!,W,N,nresids,data,spec;index=ib),xp,LBFGS(),autodiff=:forward,Optim.Options(iterations=100,show_trace=false))
    #W = inv(moment_variance(res.minimizer,gfunc!,N,nmom,nresids,data,spec))
    res = optimize(x->gmm_criterion(x,gfunc!,W,N,nresids,data,spec;index=ib),res.minimizer,Newton(),autodiff=:forward,Optim.Options(iterations=5,show_trace=false))
    #res = optimize(x->gmm_criterion(x,gfunc!,W,N,nresids,data,spec;index=ib),res.minimizer,LBFGS(),autodiff=:forward,Optim.Options(iterations=100,show_trace=false))
    #res = optimize(x->gmm_criterion(x,gfunc!,W,N,nresids,data,spec;index=ib),res.minimizer,LBFGS(),autodiff=:forward,Optim.Options(iterations=50,show_trace=false))
    Xb[:,b] = res.minimizer
end
xl = [quantile(Xb[i,:],0.10) for i in axes(Xb,1)]
xu = [quantile(Xb[i,:],0.90) for i in axes(Xb,1)]
pl = update(xl,spec,"uc")
pu = update(xu,spec,"uc")

# ======= Step 3: Write the results to a table ============ #
function write_production_table_iqr(Pl,Pu,specs,labels,outfile::String)
    form(x) = @sprintf("%0.2f",x)
    formse(x) = string("(",@sprintf("%0.2f",x),")")
    form(xl,xu) = string("[",@sprintf("%0.2f",xl),", ",@sprintf("%0.2f",xu),"]")
    nspec = length(Pl)
    
    midrule(s) = "\\cmidrule(r){$(2+(s-1)*nspec)-$(1+s*nspec)}"
    # Write the header:
    io = open(outfile, "w");
    write(io,"\\begin{tabular}{l",repeat("c",nspec*4),"}\\\\\\toprule","\n")

    # - work on elasticity parameters
    write(io," & \\multicolumn{$nspec}{c}{\$\\rho\$} & \\multicolumn{$nspec}{c}{\$\\gamma \$} & \\multicolumn{$nspec}{c}{\$\\delta_{1}\$} & \\multicolumn{$nspec}{c}{\$\\delta_{2}\$} ","\\\\\n")
    write(io,repeat(["&($s)" for s in 1:nspec],4)...,"\\\\",[midrule(s) for s in 1:4]...,"\n")
    
    # -- now write the estimates:
    
    write(io,[string("&",form(Pl[s].ρ,Pu[s].ρ)) for s in 1:nspec]...)
    write(io,[string("&",form(Pl[s].γ,Pu[s].γ)) for s in 1:nspec]...)
    write(io,[string("&",form(Pl[s].δ[1],Pu[s].δ[1])) for s in 1:nspec]...)
    write(io,[string("&",form(Pl[s].δ[2],Pu[s].δ[2])) for s in 1:nspec]...) #automate this?
    write(io,"\\\\\n")
    write(io,repeat("&",4*nspec),"\\\\\n")

    # - Write factor share parameters
    write(io," & \\multicolumn{$nspec}{c}{\$\\tilde{\\phi}_{m}\$: Mother's Time} & \\multicolumn{$nspec}{c}{\$\\tilde{\\phi}_{f}\$: Father's Time} & \\multicolumn{$nspec}{c}{\$\\tilde{\\phi}_{x}\$: Childcare} & \\multicolumn{$nspec}{c}{\$\\phi_{\\theta}\$: TFP} ","\\\\\n")
    write(io,repeat(["&($s)" for s in 1:nspec],4)...,"\\\\",[midrule(s) for s in 1:4]...,"\n")

    vlist = union([s[specvar] for s in specs, specvar in [:vm,:vf,:vm,:vθ]]...)
    for v in vlist
        if v in keys(labels)
            vname = labels[v]
            vname = string(vname) #<-?
        else
            vname = string(v)
        end
        write(io,vname)
        # write estimates
        varlist = [:βm,:βf,:βy,:βθ]
        svarlist = [:vm,:vf,:vy,:vθ]#<- 
        for k in 1:4
            var = varlist[k]
            specvar = svarlist[k]
            for j in 1:nspec
                i = findfirst(specs[j][specvar].==v)
                if isnothing(i)
                    write(io,"&","-")
                else
                    xl = getfield(Pl[j],var)[i]
                    xu = getfield(Pu[j],var)[i]
                    write(io,"&",form(xl,xu))
                end
            end
        end
        write(io,"\\\\\n")
    end 
    write(io,"\\\\\n")
    write(io,"\\bottomrule")
    write(io,"\\end{tabular}")
    close(io)
end
cluster_labels = Dict(zip(cluster_dummies,[string("Type ",string(s)[end]) for s in cluster_dummies]))
ed_labels = Dict(zip([f_ed[2:3];m_ed[2:3]],["Father some coll.","Father coll+","Mother some coll.","Mother coll+"]))
other_labels = Dict(:mar_stat => "Married",:div => "Single",:num_0_5 => "Num. of children 0-5", :constant => "Constant", :mu_k => "\$\\mu_{k}\$", :age => "Child's age", :ind02 => "Year = 2002")
labels = merge(other_labels,cluster_labels,ed_labels)

write_production_table_iqr([pl],[pu],[spec],labels,"tables/direct_estimation_iqr.tex")


break
# THEN: build data
data = child_data(panel_data,spec)
# THEN: test the update function!
xp = initial_guess(spec,"uc");
x0 = zeros(sum(length(x) for x in [spec.zc,spec.zf,spec.zg,spec.zτ])+20)
dp = update_demand_method_1(x0,spec)
x0 = [x0 ; xp]
dp,pp = update_method_1(x0,spec)
# great.

# THEN: test the moment function!
gfunc!(x,n,g,resids,data,spec) = production_demand_moments_method_1!(update_method_1(x,spec)...,n,g,resids,data)

nmom = sum([size(z,1)*!isempty(z) for z in data.Z])
nresids = length(data.Z)
g = zeros(nmom)
resids = zeros(nresids)
gfunc!(x0,2,g,resids,data,spec)
N = length(unique(panel_data.kid))
W = I(nmom)
gmm_criterion(x0,gfunc!,W,N,nresids,data,spec)
res = estimate_gmm(x0,gfunc!,W,N,length(data.Z),data,spec)

# no we can do this as two separate stages. no need to do it this way :-).
# a special one for each since βf shouldn't need divorce dummy?)
spec = (;vm,vf,vθ,vy,zc,zf,zg,zτ,vΩ,zlist_97,zlist_02,zlist_07,zlist_prod_t = [],zlist_prod = [])
data = child_data(panel_data,spec)
gfunc!(x,n,g,resids,data,spec) = demand_moments_method_1!(update_demand_method_1(x,spec),n,g,resids,data)
x0 = zeros(sum(length(x) for x in [spec.zc,spec.zf,spec.zg,spec.zτ])+20)

nmom = sum([size(z,1)*!isempty(z) for z in data.Z])
nresids = length(data.Z)
g = zeros(nmom)
resids = zeros(nresids)
gfunc!(x0,2,g,resids,data,spec)
N = length(unique(panel_data.kid))
W = I(nmom)
gmm_criterion(x0,gfunc!,W,N,nresids,data,spec)
res = estimate_gmm(x0,gfunc!,W,N,length(data.Z),data,spec)
dp = update_demand_method_1(res.est,spec)


spec = (;vm,vf,vθ,vy,zc,zf,zg,zτ,vΩ,zlist_97=[],zlist_02=[],zlist_07=[],zlist_prod_t,zlist_prod)

# THEN: build data
data = child_data(panel_data,spec)

gfunc!(x,n,g,resids,data,spec) = production_moments_method_1!(dp,update(x,spec,"uc"),n,g,resids,data)

nmom = sum([size(z,1)*!isempty(z) for z in data.Z])
nresids = length(data.Z)
g = zeros(nmom)
resids = zeros(nresids)
gfunc!(x0,2,g,resids,data,spec)
N = length(unique(panel_data.kid))
W = I(nmom)

# impose box constraints!!
lower = [-10;-10;fill(-Inf,length(xp)-2)]
upper = [1.;1.;fill(Inf,length(xp)-2)]

res = optimize(x->gmm_criterion(x,gfunc!,W,N,nresids,data,spec),lower,upper,xp,Fminbox(LBFGS()),autodiff=:forward,Optim.Options(iterations=100,show_trace=true))


res2 = optimize(x->gmm_criterion(x,gfunc!,W,N,nresids,data,spec),res.minimizer,Newton(),autodiff=:forward,Optim.Options(show_trace=true))

Ω = moment_variance(res.minimizer,gfunc!,N,nmom,nresids,data,spec)
W = inv(Ω)

Ω = moment_variance(res2.minimizer,gfunc!,N,nmom,nresids,data,spec)
W = inv(Ω)


res3 = optimize(x->gmm_criterion(x,gfunc!,W,N,nresids,data,spec),res2.minimizer,Newton(),autodiff=:forward,Optim.Options(iterations=40,show_trace=true))

Xb = bootstrap_gmm(res3.minimizer,gfunc!,W,N,nresids,data,spec)

Xb2 = bootstrap_gmm(xp,gfunc!,W,N,nresids,data,spec)

v = cov(Xb')
se = sqrt.(diag(v))

# --- concentrate out ρ and γ for identification purposes ---- #

ρgrid = [-10.,-7.5,-5.,-2.5,0.9]
γgrid = [-10.,-7.5,-5.,-2.5,0.9]

Q = zeros(5)
Ql = zeros(5)
Qu = zeros(5)
for i in eachindex(Q)
    @show i
    gfunc!(x,n,g,resids,data,spec) = production_moments_method_1!(dp,update([ρgrid[i];-5.;x],spec,"uc"),n,g,resids,data)
    res = optimize(x->gmm_criterion(x,gfunc!,W,N,nresids,data,spec),xp[3:end],Newton(),autodiff=:forward,Optim.Options(iterations=30,show_trace=true))
    Q[i] = res.minimum
    Qb = zeros(50)
    for b in 1:50
        @show i b
        ib = rand(1:N,N)
        resb = optimize(x->gmm_criterion(x,gfunc!,W,N,nresids,data,spec;index=ib),res.minimizer,Newton(),autodiff=:forward,Optim.Options(iterations=5))
        Qb[b] = resb.minimum
    end
    Ql[i] = quantile(Qb,0.05)
    Qu[i] = quantile(Qb,0.95)
end

plot!(ρgrid , Q,color="blue")
plot!(ρgrid, Ql,color="red")
plot!(ρgrid, Qu,color="red")


ρgrid = [-7.5,-2.5,0.9]

ρgrid = [-10.,-7.5,-5.,-2.5,0.9]
γgrid = [-7.5,-2.5,0.9]
Q = zeros(5,3)
Ql = zeros(5,3)
Qu = zeros(5,3)

for i in axes(Q,1), j in axes(Q,2)
    @show i j
    gfunc!(x,n,g,resids,data,spec) = production_moments_method_1!(dp,update([ρgrid[i];γgrid[j];x],spec,"uc"),n,g,resids,data)
    res = optimize(x->gmm_criterion(x,gfunc!,W,N,nresids,data,spec),xp[3:end],Newton(),autodiff=:forward,Optim.Options(iterations=30,show_trace=true))

    Qb = zeros(50)
    for b in 1:50
        @show i b
        ib = rand(1:N,N)
        resb = optimize(x->gmm_criterion(x,gfunc!,W,N,nresids,data,spec;index=ib),res.minimizer,Newton(),autodiff=:forward,Optim.Options(iterations=10))
        Qb[b] = resb.minimum
    end

    Q[i,j] = res.minimum
    Ql[i,j] = quantile(Qb,0.025)
    Qu[i,j] = quantile(Qb,0.975)
end

using Plots
#plot((plot(ρgrid,Q[:,j]) for j in 1:3)...)
plot(ρgrid,Q[:,1],color="blue")
plot!(ρgrid,Ql[:,1],color="red")
plot!(ρgrid,Qu[:,1],color="red")
g = plot()
for i in 1:3
    plot!(g,ρgrid,Q[:,i],color="blue")
    plot!(g,ρgrid,Ql[:,i],color="red")
    plot!(g,ρgrid,Qu[:,i],color="red")
end  
g  

# NOTE: this simulation is wrong, have to re-minimize over the other parameters, too. Will take a bit longer!!

plot(ρgrid,Q[:,1],color="blue",label="γ=-7.5")
plot!(ρgrid,Ql[:,1],color="blue",linestyle=:dash,label=false)
plot!(ρgrid,Qu[:,1],color="blue",linestyle=:dash,label=false)

plot!(ρgrid,Q[:,2],color="red",label="γ=-2.5")
plot!(ρgrid,Ql[:,2],color="red",linestyle=:dash,label=false)
plot!(ρgrid,Qu[:,2],color="red",linestyle=:dash,label=false)

plot!(ρgrid,Q[:,3],color="green",label="γ=0.9")
plot!(ρgrid,Ql[:,3],color="green",linestyle=:dash,label=false)
plot!(ρgrid,Qu[:,3],color="green",linestyle=:dash,label=false)
xlabel!("ρ")


# ===== Now let's run a dodgy bootstrap ======== #
# this might actually be the best way to show everything. Much cheaper.

# 

Xb = zeros(length(xp),50)
gfunc!(x,n,g,resids,data,spec) = production_moments_method_1!(dp,update(x,spec,"uc"),n,g,resids,data)

for b in 1:50
    @show b
    ib = rand(1:N,N)
    resb = optimize(x->gmm_criterion(x,gfunc!,W,N,nresids,data,spec;index=ib),xp,Newton(),autodiff=:forward,Optim.Options(iterations=20))
    resb = optimize(x->gmm_criterion(x,gfunc!,W,N,nresids,data,spec;index=ib),resb.minimizer,LBFGS(),autodiff=:forward,Optim.Options(iterations=50,show_trace=true))
    Xb[:,b] = resb.minimizer
end

# ===== Alt Estimation Routine ======= #

# basically a failure, but we should run the boostrap with two weighting matrices just to be good?

gfunc!(x,n,g,resids,data,spec) = production_moments_method_1!(dp,update([-0.5;-0.5;x],spec,"uc"),n,g,resids,data)
res = optimize(x->gmm_criterion(x,gfunc!,W,N,nresids,data,spec),xp[3:end],Newton(),autodiff=:forward,Optim.Options(iterations=5,show_trace=true))
Ω = moment_variance(res.minimizer,gfunc!,N,nmom,nresids,data,spec)
W = inv(Ω)
res2 = optimize(x->gmm_criterion(x,gfunc!,W,N,nresids,data,spec),res.minimizer,Newton(),autodiff=:forward,Optim.Options(iterations=20,show_trace=true))
Ω = moment_variance(res2.minimizer,gfunc!,N,nmom,nresids,data,spec)
W = inv(Ω)


gfunc!(x,n,g,resids,data,spec) = production_moments_method_1!(dp,update(x,spec,"uc"),n,g,resids,data)
x0 = [-0.5;-0.5;res2.minimizer]
res3 = optimize(x->gmm_criterion(x,gfunc!,W,N,nresids,data,spec),x0,Newton(),autodiff=:forward,Optim.Options(iterations=50,show_trace=true))
Ω = moment_variance(res3.minimizer,gfunc!,N,nmom,nresids,data,spec)
W = inv(Ω)
res4 = optimize(x->gmm_criterion(x,gfunc!,W,N,nresids,data,spec),x0,Newton(),autodiff=:forward,Optim.Options(iterations=50,show_trace=true))

# ===== Estimation with ρ and γ fixed ===== #
gfunc!(x,n,g,resids,data,spec) = production_moments_method_1!(dp,update([-2.;-2.;x],spec,"uc"),n,g,resids,data)
W = I(nmom)
res = estimate_gmm(xp[3:end],gfunc!,W,N,nresids,data,spec)
# this doesn't work very well.

