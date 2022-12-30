# this script loads the data and creates:
#   (1) a named tuple called "data"
#   (2) A vector of instruments for the production moments Zprod that are used always
#   (3) A vector of instruments for the production moments for married couples, ZprodF

# the functions that we call to estimate the gmm routine use the data object and take as given that it will contain correctly named variables and objects used in estimation.


# Step 1: create the data object
D = DataFrame(CSV.File("data/gmm_full_vertical.csv",missingstring = "NA"))
d = DataFrame(CSV.File("data/gmm_full_horizontal.csv",missingstring = "NA"))
NT = size(D)[1]
N = size(d)[1]


# vertical variables
price_g = convert(Array{Float64,1},coalesce.(D.price_g,0))
logprice_g = log.(price_g)
price_c = convert(Array{Float64,1},coalesce.(D.p_4f,0))
logprice_c = log.(price_c)
wage_m = convert(Array{Float64,1},coalesce.(D.m_wage,1))
logwage_m = log.(wage_m)
wage_f = convert(Array{Float64,1},coalesce.(D.f_wage,1))
logwage_f = log.(wage_f)
log_mtime = convert(Array{Float64,1},coalesce.(D.log_mtime,-1))
log_ftime = convert(Array{Float64,1},coalesce.(D.log_ftime,-1))
log_good = convert(Array{Float64,1},coalesce.(D.log_good,-1))
log_chcare = convert(Array{Float64,1},coalesce.(D.log_chcare,-1))
lY = convert(Array{Float64,1},coalesce.(D.lY,-1))
d.f_ed = coalesce.(d.f_ed,"--")
num_0_5_long = convert(Array{Float64,1},coalesce.(D.num_0_5,0))

# horizontal variables
age = convert(Array{Int64,1},d.age)
mar = convert(Array{Int64,1},d.mar_stable)
m_white = convert(Array{Float64,1},coalesce.(d.Race,0).==1)
num_0_5 = convert(Array{Float64,1},d.num_0_5)
all_prices = convert(Array{Int64,1},(d.all_prices) .& (d.age.<=8) .& (.!ismissing.(d.AP97)) .& (.!ismissing.(d.ltau_m97)) .& (.!ismissing.(d.AP02)) .& (.!ismissing.(d.LW97))) #<- try it and see
A97 = convert(Array{Float64,1},coalesce.((d.AP97 .- mean(d.AP97[all_prices.==1]))/15,0))
A02 = convert(Array{Float64,1},coalesce.((d.AP02 .- mean(d.AP02[all_prices.==1]))/15,0))
L97 = convert(Array{Float64,1},coalesce.((d.LW97 .- mean(d.LW97[all_prices.==1]))/15,0))
L02 = convert(Array{Float64,1},coalesce.((d.LW02 .- mean(d.LW02[all_prices.==1]))/15,0))
ltau_m97 = convert(Array{Float64,1},coalesce.(d.ltau_m97,0)) #<- need missing time indicators?
ltau_m02 = convert(Array{Float64,1},coalesce.(d.ltau_m02,0))
ltau_f97 = convert(Array{Float64,1},coalesce.(d.ltau_f97,0))
ltau_f02 = convert(Array{Float64,1},coalesce.(d.ltau_f02,0))
log_good02 = convert(Array{Float64,1},coalesce.(D.log_good[D.year.==2002],0))
log_chcare97 = convert(Array{Float64,1},coalesce.(D.log_chcare[D.year.==1997],0))

# missing indicators
good_missing = convert(Array{Int64,1},D.good_missing)
mtime_missing = convert(Array{Int64,1},D.mtime_missing)
chcare_missing = convert(Array{Int64,1},D.chcare_missing)
ftime_missing = convert(Array{Int64,1},D.ftime_missing)
price_missing = convert(Array{Int64,1},D.price_missing)
# -- education variables
m_ed = zeros(Int64,N);
m_ed[d.m_ed.=="<12"] .= 1
m_ed[d.m_ed.=="12"] .= 1
m_ed[d.m_ed.=="13-15"] .= 2
m_ed[d.m_ed.=="16"] .= 3
m_ed[d.m_ed.==">16"] .= 3
f_ed = zeros(Int64,N);
f_ed[d.f_ed.=="<12"] .= 1
f_ed[d.f_ed.=="12"] .= 1
f_ed[d.f_ed.=="13-15"] .= 2
f_ed[d.f_ed.=="16"] .= 3
f_ed[d.f_ed.==">16"] .= 3
#-------
age_dummies = age.==(3:8)' #
m_ed_dummies = m_ed.==(2:3)'
f_ed_dummies = f_ed.==(2:3)'
Xg = [mar (1 .-mar) m_ed_dummies f_ed_dummies age] #
Xf = [ones(N) f_ed_dummies age] #
Xm = [mar (1 .-mar) m_ed_dummies age] #
#Xθ = [m_ed_dummies num_0_5] # 6
Xθ = [ones(N) age mar m_ed_dummies f_ed_dummies] #[m_ed_dummies num_0_5 mar] # num_0_5] # 6
Xθf = [ones(N) age m_ed_dummies f_ed_dummies]
# add full vector??

# ------------- instruments ------------------------- #
I97 = 1:6:NT
I99 = 3:6:NT
I01 = 5:6:NT
Xg_long = [kron(Xg,ones(6)) num_0_5_long]
Xg_long_m = Xg_long[:,[1;3:end]]
Xm_long = [kron(Xm,ones(6)) num_0_5_long]
Xm_long_s = Xm_long[:,2:end] #<- drop the married dummy
Xm_long_m = Xm_long[:,[1;3:end]]
Xf_long = [kron(Xf,ones(6)) num_0_5_long]
prices97 = [logprice_c[I97] logprice_g[I97] logwage_m[I97]] # logwage_f[I97]]
prices99 = [logprice_c[I99] logprice_g[I99] logwage_m[I99] logwage_f[I99]]
prices01 = [logprice_c[I01] logprice_g[I01] logwage_m[I01] logwage_f[I01]]

# -- achievement equation instruments
# -- preferred instruments use everything
Z_prod = ([Xθ ltau_m02 ltau_f02 logprice_g[I97] L97],[Xθ ltau_m02 ltau_f02 logprice_g[I97] A97]) #<- taking out prices
Z_prodF = ([Xθf ltau_m02 ltau_f02 logprice_g[I97] L97],[Xθf ltau_m02 ltau_f02 logprice_g[I97] A97])
# NOTE: we can add log_good02 as an instrument with minimal effect on estimates

#Z_prod = [ones(N) Xθ age lY[I97] logprice_g[I97] logwage_m[I97] convert(Array{Float64,1},coalesce.((d.LW97 .- 100)/15,0))]

# -- share equation instruments
Zm = ([Xm_long_m log.(wage_m./price_g)],[Xf_long log.(wage_f./price_g)],[Xg_long_m log.(price_c./price_g)],[Xg_long_m log.(wage_f./wage_m)],[Xg_long_m log.(price_c./wage_m)])
Km = [size(Zm[i])[2] for i=1:5]
Zs = ([Xm_long_s log.(wage_m./price_g)],[Xm_long_s log.(price_c./price_g)],[Xm_long_s log.(price_c./wage_m)])
Ks = [size(Zs[i])[2] for i=1:3] #<- this gives the number of instruments for the three sets of share equations

# create the data object:
data = (age = age,mar = mar,all_prices = all_prices,A97=A97,A02=A02,L97=L97,L02=L02,price_g = price_g,logprice_g=logprice_g,price_c=price_c,logprice_c=logprice_c,wage_m=wage_m,
logwage_m=logwage_m,wage_f=wage_f,logwage_f=logwage_f,log_mtime=log_mtime,log_ftime=log_ftime,log_good=log_good,log_chcare=log_chcare,lY=lY,good_missing=good_missing,
mtime_missing=mtime_missing,chcare_missing=chcare_missing,ftime_missing=ftime_missing,price_missing=price_missing,Zs=Zs,
Zm=Zm,Ks=Ks,Km=Km,Xm = Xm_long,Xf = Xf_long, Xg = Xg_long, Xθ = Xθ,Xθf=Xθf,ltau_m97=ltau_m97,ltau_f97=ltau_f97)
