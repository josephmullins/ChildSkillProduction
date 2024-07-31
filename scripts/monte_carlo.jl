# a script to pick parameters for the monte-carlo simulation

include("../src/model.jl")
include("../src/model_older_children.jl")
include("../src/estimation.jl")

# =======================   read in the data and estimates ===================================== #
# - load the data 
panel_data = DataFrame(CSV.File("../../../PSID_CDS/data-derived/psid_fam.csv",missingstring = ["","NA"]))
panel_data, m_ed, f_ed = prep_data(panel_data)
panel_data = DataFrame(filter(x-> sum(skipmissing(x.ind_not_sample.==0))>0 || sum(x.all_prices)>0,groupby(panel_data,:kid)))

wage_types = DataFrame(CSV.File("data/wage_types.csv"))

panel_data=innerjoin(panel_data, wage_types, on = :MID) #merging in cluster types
cluster_dummies=make_dummy(panel_data,:cluster) #cluster dummies made

# get the four specifications we settle on in the paper
spec1,spec2,spec3,spec4 = get_specifications(m_ed,f_ed,cluster_dummies)

# read in estimates from spec 3
est = readdlm("output/est_nbs_spec3")[:]

p = update(est,spec3,"nbs")

# ============ Calibrate Parameters ====================================== #
# (1) residual variation of child care prices relative to mother's wage
# (2) covariance of demand residuals across time periods
# (3) variance of demand residuals
# (4) variance of residual in outcome equation
# (5) variance of the mother's time input (logged, conditional on Zθ)
N = length(unique(panel_data.kid))
data = child_data(panel_data,spec3)
R = zeros(30,N)
for n in 1:N
    @views demand_residuals_all!(R[1:9,n],p,n,data)
    @views production_residuals_all!(R[10:end,n],p,p,n,data)
end


r_keep = sum(R[1:2,:].!=0,dims=1)[:].>1 #<- keep only non-missing residuals in both years
σζ1 = sqrt(cov(R[1,r_keep],R[2,r_keep]))
σζ2 = sqrt(var(R[1,r_keep]) - σζ1^2)
σξ = std(R[10,:]) / 2 #<- we divide by 2 to increase the starkness of comparisons.

I_keep = (panel_data.year.==1997) .& .!ismissing.(panel_data.logprice_c_m) .& .!ismissing.(panel_data.div) .& .!ismissing.(panel_data.log_mtime)
X = Matrix{Float64}(panel_data[I_keep,spec3.vθ[1:end-1]])
Y1 = Vector{Float64}(panel_data.logprice_c_m[I_keep])
Y2 = Vector{Float64}(panel_data.log_mtime[I_keep])

σx = std(Y2 .- X*inv(X' * X) * X' * Y2)
σπ = std(Y1 .- X*inv(X' * X) * X' * Y1)

p = (;ρ = -3.,a = 0.5,δ = 0.1, σζ1, σζ2, σπ, σξ, σx)

# ================= Functions to run the simulation ====================== #
# main dgp
function gen_data(p,N)
    (;ρ, a, δ, σξ, σπ, σζ1,σζ2, σx) = p
    rel_price = rand(LogNormal(0,σπ),N)
    ζ1 = rand(Normal(0.,σζ1),N) #<- true variation in x2
    ζ2 = rand(Normal(0.,σζ2),N) #<- measurement error in x2
    x1 = rand(LogNormal(0.,σx),N)
    x2 = (a/(1-a))^(1/(ρ-1)) .* rel_price.^(1/(ρ-1)) .* x1 .* exp.(ζ1)
    logy = δ * log.( (a .* x1 .^ ρ .+ (1 - a) .* x2 .^ ρ ) .^ (1/ρ) ) .+ rand(Normal(0,σξ),N)
    logx2_obs = log.(x2) .+ ζ2
    return (;logy,x1,x2,logx2_obs,rel_price)
end
# dgp with independent variation
function gen_data2(p,N)
    (;ρ, a, δ, σξ, σπ, σζ1,σζ2, σx) = p
    #     x2 = (a/(1-a))^(1/(ρ-1)) .* rel_price.^(1/(ρ-1)) .* x1 .* exp.(ζ1)
    sdlogx2 = sqrt((1/(ρ-1))^2*σπ^2 + σx^2 + σζ1^2)
    rel_price = rand(LogNormal(0,σπ),N)
    ζ2 = rand(Normal(0.,σζ2),N) #<- measurement error in x2
    x1 = rand(LogNormal(0.,σx),N)
    x2 = rand(LogNormal(0.,sdlogx2),N)
    logy = δ * log.( (a .* x1 .^ ρ .+ (1 - a) .* x2 .^ ρ ) .^ (1/ρ) ) .+ rand(Normal(0,σξ),N)
    logx2_obs = log.(x2) .+ ζ2
    return (;logy,x1,x2,logx2_obs,rel_price)
end


# this function works for both estimators 1 and 2.
function Q1_nlls(p,data)
    (;logy,x1,x2) = data
    (;ρ,a,δ) = p
    r2 = logy .- δ * log.( (a .* x1 .^ ρ .+ (1 - a) .* x2 .^ ρ) .^ (1/ρ) )
    return sum(r2.^2)
end

# this function works for estimators 3 and 4
function Q2_nlls(p1,p2,data)
    # p1 are the perceived parameters of technology
    # p2 are the true parameters of technology
    (;logy,x1,logx2_obs,rel_price) = data
    (;δ) = p2
    r1 = logx2_obs .- log.(x1) .- (1/(p1.ρ-1))*log(p1.a/(1-p1.a)) .- (1/(p1.ρ-1))*log.(rel_price)
    x2 = (p1.a/(1-p1.a))^(1/(p1.ρ-1)) .* rel_price.^(1/(p1.ρ-1)) .* x1
    r2 = logy .- δ * log.( (p2.a .* x1 .^ p2.ρ .+ (1 - p2.a) .* x2 .^ p2.ρ) .^ (1/p2.ρ) )
    return sum(r1.^2) + sum(r2.^2)
end

function monte_carlo(N,B,p)
    ρb = zeros(4,B)
    ab = zeros(4,B)
    δb = zeros(4,B)
    for b in 1:B
        println("Doing round $b of $B trials")
        dat = gen_data(p,N)
        lower = [-50.,0.,0.]
        upper = [1.,1.,1.]
        x0 = [p.ρ, p.a, p.δ]
        #res1 = optimize(x->Q1_nlls((;p...,ρ=x),dat),-50.,1.) #[-2.]) #-300,1.) 
        res2 = optimize(x->Q1_nlls((;ρ=x[1],a=x[2],δ = x[3]),dat),lower,upper,x0,Fminbox(LBFGS()),autodiff=:forward)
        res4 = optimize(x->Q2_nlls((;ρ=x[1],a=x[2]),(;ρ=x[1],a=x[2],δ = x[3]),dat),lower,upper,x0,Fminbox(LBFGS()),autodiff=:forward) 
        lower = [-50.,0.,-50.,0.,0.]
        upper = [1.,1.,1.,1.,1.]
        x0 = [p.ρ, p.a, p.ρ, p.a, p.δ]
        res3 = optimize(x->Q2_nlls((;ρ=x[1],a=x[2]),(;ρ=x[3],a=x[4],δ = x[5]),dat),lower,upper,x0,Fminbox(LBFGS()),autodiff=:forward)
        # lower = [0.,0.]
        # upper = [1.,1.]
        # x0 = [p.a, p.δ]
        # res5 = optimize(x->Q1_nlls((;ρ=p.ρ,a=x[1],δ = x[2]),dat),lower,upper,x0,Fminbox(LBFGS()),autodiff=:forward)
        dat = gen_data2(p,N)
        lower = [-50.,0.,0.]
        upper = [1.,1.,1.]
        x0 = [p.ρ, p.a, p.δ]
        res5 = optimize(x->Q1_nlls((;ρ=x[1],a=x[2],δ = x[3]),dat),lower,upper,x0,Fminbox(LBFGS()),autodiff=:forward)

        ρb[:,b] .= (res2.minimizer[1],res3.minimizer[3],res4.minimizer[1],res5.minimizer[1])
        ab[:,b] .= (res2.minimizer[2],res3.minimizer[4],res4.minimizer[2],res5.minimizer[2])
        δb[:,b] .= (res2.minimizer[3],res3.minimizer[5],res4.minimizer[3],res5.minimizer[3])
    end
    return ρb,ab,δb
end

# ================== Results ========================== #

N_vec = [500,1_000,5_000]

bias = zeros(4,3,3)
sd = zeros(4,3,3)

Random.seed!(73124)
for j in 1:3
    ρb,ab,δb = monte_carlo(N_vec[j],500,p)
    bias[:,j,1] .= p.ρ .- mean(ρb,dims=2)[:]
    sd[:,j,1] .= std(ρb,dims=2)[:]
    bias[:,j,2] .=  p.a .- mean(ab,dims=2)[:]
    sd[:,j,2] .= std(ab,dims=2)[:]
    bias[:,j,3] .=  p.δ .- mean(δb,dims=2)[:]
    sd[:,j,3] .= std(δb,dims=2)[:]
end

function write_monte_carlo_table(sd,bias,N_vec,outfile::String)
    form(x) = @sprintf("%0.2f",x)
    io = open(outfile, "w");
    par_string = ["\$\\rho\$","\$a\$","\$\\delta\$"]
    write(io,"\\begin{tabular}{lcccccccc} \\\\\\toprule","\n")
    for i in 1:3
        write(io," & \\multicolumn{8}{c}{Results for ",par_string[i],"} \\\\ \n")
        write(io," & \\multicolumn{4}{c}{Bias} & \\multicolumn{4}{c}{Std. Dev.} \\\\ \n")
        write(io,("& ($j)" for b in 1:2 for j in 1:4)...,"\\\\ \n")
        write(io,"\\cmidrule(r){2-5}\\cmidrule(r){6-9}")
        for n in eachindex(N_vec)
            write(io,"\$N\$ = $(N_vec[n])")
            for j in 1:4
                write(io," & ",form(bias[j,n,i]))
            end
            for j in 1:4
                write(io," & ",form(sd[j,n,i]))
            end
            write(io,"\\\\ \n")
        end
        if i<3
            write(io,"&&&&&&&& \\\\ \n")
        end
    end
    write(io,"\\bottomrule")
    write(io,"\\end{tabular}")
    close(io)
end

write_monte_carlo_table(sd,bias,N_vec,"tables/monte_carlo_results.tex")

