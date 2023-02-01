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

function get_wage_data(data,vlist::Array{Symbol,1},fe)
    N = size(data)[1]
    data=select(data, [:MID;:logwage_m;vl])
    d=data[completecases(data), :]
    lW=Vector{Float64}(d[!,:logwage_m]) #return non-demeans
    if fe
        d=groupby(d,:MID)
        d=transform(d, :logwage_m => mean) 
        d[!,:logwage_m_demean] = d.logwage_m-d.logwage_m_mean #overwrites log_wage with the de-meaned
        lW=Vector{Float64}(d[!,:logwage_m_demean])
        c=d #need to preserve original dataframe to return non-demeaned variables (where do we actually used the demeaned vlist)

        for i in 1:length(vlist)
            d=groupby(d,:MID)
            d=transform(d, vlist[i] => mean) #creates a new mean column
            d[!,vlist[i]] = d[!,vlist[i]]-d[!,end] #subtracts off the newest column, which should be the newly constructed mean
        end
    end
    return lW,Matrix{Float64}(c[!,vlist]),d
end

#ed_dummies=make_dummy(D,:m_ed) #education dummies made
#vl=[ed_dummies;:age_mother]

# a function to return regression coefficents and *also* residuals
# -- also return an estimate of individual fixed effects if fe=true (a mean of residuals for each MID)
function wage_regression(data,vlist,fe)
    if fe
        lW,X,d = get_wage_data(data,vlist,fe) #using the demeaned values
        coef=inv(X'X)*X'lW
        d.resid = missings(Float64, nrow(d))
        tcoef=transpose(coef)
            for i in 1:nrow(d)
                d.resid[i]=tcoef*X[i,:]
            end    
        d.resid=d.logwage_m-d.resid #residuals from non-demeaned
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
        d.resid=lW-d.resid #herelW is just the logwage normally
        return Vector{Float64}(coef),Vector{Float64}(d.resid) 
    end
end

function wage_clustering(wage_reg,fe)
    if fe
        df=wage_reg[3]
        dat=df[:,:resid_mean]
        dat=unique(dat)
        features=collect(Vector{Float64}(dat)')
        result = kmeans(features, 5; maxiter=100, display=:iter)
        a=assignments(result)
        centers=result.centers
        clusters=DataFrame(MID=unique(df.MID),cluster=a)
    else
        df=wage_reg[2]
        dat=df[:,:resid_mean]
        dat=unique(dat)
        features=collect(Vector{Float64}(dat)')
        result = kmeans(features, 5; maxiter=100, display=:iter)
        a=assignments(result)
        centers=result.centers
        clusters=DataFrame(MID=unique(df.MID),cluster=a)
    end
    return clusters,centers
end

function generate_cluster_assignment(dat,fe)
    D=dat
    D[!,:logwage_m] = log.(D.m_wage)
    D[!,:age_sq] = D.age_mother.^2
    ed_dummies=make_dummy(D,:m_ed) 
    vl=[ed_dummies;:age_mother]

    df=wage_regression(D,vl,fe)
    cluster_assignment=wage_clustering(df,fe)

    return cluster_assignment
end

output=generate_cluster_assignment(D,true)

