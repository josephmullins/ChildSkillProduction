using Optim
using ForwardDiff
using Statistics
using CSV
using DataFrames

# read in data:

D = DataFrame(CSV.File("../../../PSID_CDS/data-derived/MotherPanelCDS.csv",missingstring = "NA"))
D[!,:logwage_m] = log.(D.m_wage)
D[!,:age_sq] = D.age_mother.^2

#groupby(D)

function get_wage_data(data,vlist::Array{Symbol,1},fe=false)
    N = size(data)[1]
    index_select = .!ismissing.(data.logwage_m)
    for v in vlist
        index_select = index_select .& (.!ismissing.(data[!,v]))
    end
    if fe
        # de-mean all variables by individual (MID)
        # use groupby <- "split-apply-combine"
    end
    return Vector{Float64}(data[index_select,:logwage_m]),Matrix{Float64}(data[index_select,vlist])
end

function make_dummy(data,var::Symbol)
    vals = unique(skipmissing(data[!,var]))
    nvals = length(vals)
    vnames = [Symbol(var,"_",x) for x in vals]
    for i in 1:nvals
        data[!,vnames[i]] = data[!,var].==vals[i]
    end
    return vnames
end

# a function to return regression coefficents and *also* residuals
# -- also return an estimate of individual fixed effects if fe=true (a mean of residuals for each MID)
function wage_regression(data,vlist,fe=false)
    
end

lW,X = get_wage_data(D,[ed_dummies;:age_mother])
# regression formula for coefficients
inv(X'X)*X'lW

# with residuals, use package Clustering.jl to perform K-means on the estimates of fixed effects

