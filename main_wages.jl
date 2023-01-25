using Optim
using ForwardDiff
using Statistics
using CSV
using DataFrames
using Plots
using Clustering

# read in data:

/Users/madisonbozich/MotherPanelCDS.csv

D = DataFrame(CSV.File("../../../PSID_CDS/data-derived/MotherPanelCDS.csv",missingstring = "NA"))
D = DataFrame(CSV.File("/Users/madisonbozich/MotherPanelCDS.csv",missingstring = "NA"))
D[!,:logwage_m] = log.(D.m_wage)
D[!,:age_sq] = D.age_mother.^2

#groupby(D)

function make_dummy(data,var::Symbol)
    vals = unique(skipmissing(data[!,var]))
    nvals = length(vals)
    vnames = [Symbol(var,"_",x) for x in vals]
    for i in 1:nvals
        data[!,vnames[i]] = data[!,var].==vals[i]
    end
    return vnames
end


# dropmissing()
function get_wage_data(data,vlist::Array{Symbol,1},fe=false)
    N = size(data)[1]
    index_select = .!ismissing.(data.logwage_m)
    for v in vlist
        index_select = index_select .& (.!ismissing.(data[!,v]))
    end
    if fe
        # de-mean all variables by individual (MID)
        # use groupby <- "split-apply-combine"
        d=data[index_select,:]
        d=groupby(d,:MID)
        d=transform(d, :logwage_m => mean) 
        lW = d.logwage_m-d.logwage_m_mean #overwrites log_wage with the de-meaned
        
        #de-means the log wages, did not de-mean age or the booleans
    end
    return Vector{Float64}(lW),Matrix{Float64}(data[index_select,vlist]),data[index_select,:]
end

ed_dummies=make_dummy(D,:m_ed) #education dummies made
vl=[ed_dummies;:age_mother]

# a function to return regression coefficents and *also* residuals
# -- also return an estimate of individual fixed effects if fe=true (a mean of residuals for each MID)
function wage_regression(data,vlist,fe)
    
    if fe

        lW,X,d = get_wage_data(data,vlist,fe)
        coef=inv(X'X)*X'lW

        d.resid = missings(Float64, nrow(d))
        tcoef=transpose(coef)
    
        for i in 1:nrow(d)
            d.resid[i]=tcoef*X[i,:]
        end    

        d.resid=d.logwage_m-d.resid

        d=groupby(d,:MID)
        d=transform(d, :resid => mean) #a row of the means repeating for each group; if not desired form use select

        return Vector{Float64}(coef),Vector{Float64}(d.resid),d

    else

        lW,X,d = get_wage_data(data,vlist,fe)
        coef=inv(X'X)*X'lW

        d.resid = missings(Float64, nrow(d))
        tcoef=transpose(coef)
    
        for i in 1:nrow(d)
            d.resid[i]=tcoef*X[i,:]
        end    

        d.resid=lW-d.resid

        return Vector{Float64}(coef),Vector{Float64}(d.resid) 
    end
end

df=wage_regression(D,vl,true)

####some test statements before I clusteredtest=D[!, [:MID,:logwage_m,:m_ed_12,:m_ed_16]]
#test1=test[completecases(test),:]
#test2=groupby(test1,:MID)
#test3=transform(test2, :logwage_m => mean)
#test3[!,:logwage_m] = test3.logwage_m-test3.logwage_m_mean

#ed_dummies=make_dummy(D,:m_ed) #education dummies made
#vl=[ed_dummies;:age_mother]
#lW,X,d = get_wage_data(D,vl,true)


# with residuals, use package Clustering.jl to perform K-means on the estimates of fixed effects

#below I clustered data based upon the fixed effects and variable list (education dummies and age)

vl=[ed_dummies;:age_mother]
df=df[3]
df=df[:,[vl;:resid_mean]]
features=collect(Matrix{Float64}(df)')

result = kmeans(features, 5; maxiter=100, display=:iter)

nclusters(result)

a=assignments(result)
c = counts(result) # get the cluster sizes
M = R.centers
