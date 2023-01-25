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


<<<<<<< HEAD
function get_wage_data(data,vlist::Array{Symbol,1},fe)
=======
# dropmissing()
function get_wage_data(data,vlist::Array{Symbol,1},fe=false)
>>>>>>> b9ea32a4280baaa33fc1799564b8c26de3de6e7b
    N = size(data)[1]
    index_select = .!ismissing.(data.logwage_m)
    for v in vlist
        index_select = index_select .& (.!ismissing.(data[!,v]))
    end
<<<<<<< HEAD
    d=data[index_select,:]
    lW=Vector{Float64}(d[!,:logwage_m]) #return non-demeans
=======
>>>>>>> b9ea32a4280baaa33fc1799564b8c26de3de6e7b
    if fe
        # de-mean all variables by individual (MID)
        # use groupby <- "split-apply-combine"
        d=data[index_select,:]
        d=groupby(d,:MID)
        d=transform(d, :logwage_m => mean) 
<<<<<<< HEAD
        d[!,:logwage_m_demean] = d.logwage_m-d.logwage_m_mean #overwrites log_wage with the de-meaned
        lW=Vector{Float64}(d[!,:logwage_m_demean])

        for i in 1:length(vlist)
            d=groupby(d,:MID)
            d=transform(d, vlist[i] => mean) #creates a new mean column
            d[!,vlist[i]] = d[!,vlist[i]]-d[!,end] #subtracts off the newest column, which should be the newly constructed mean
        end
=======
        lW = d.logwage_m-d.logwage_m_mean #overwrites log_wage with the de-meaned
        
        #de-means the log wages, did not de-mean age or the booleans
>>>>>>> b9ea32a4280baaa33fc1799564b8c26de3de6e7b
    end
<<<<<<< HEAD
    return lW,Matrix{Float64}(data[index_select,vlist]),d
=======
    return Vector{Float64}(lW),Matrix{Float64}(data[index_select,vlist]),data[index_select,:]
>>>>>>> b9ea32a4280baaa33fc1799564b8c26de3de6e7b
end

ed_dummies=make_dummy(D,:m_ed) #education dummies made
vl=[ed_dummies;:age_mother]

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

<<<<<<< HEAD
        d.resid=d.logwage_m-d.resid #residuals from non-demeaned
=======
        d.resid=d.logwage_m-d.resid
>>>>>>> b9ea32a4280baaa33fc1799564b8c26de3de6e7b

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



<<<<<<< HEAD

lW,X,d = get_wage_data(D,vl,true)
df=wage_regression(D,vl,true)
=======
#ed_dummies=make_dummy(D,:m_ed) #education dummies made
#vl=[ed_dummies;:age_mother]
#lW,X,d = get_wage_data(D,vl,true)
>>>>>>> b9ea32a4280baaa33fc1799564b8c26de3de6e7b

<<<<<<< HEAD
=======

# with residuals, use package Clustering.jl to perform K-means on the estimates of fixed effects
>>>>>>> b9ea32a4280baaa33fc1799564b8c26de3de6e7b


<<<<<<< HEAD
function wage_clustering(wage_reg,fe)
    if fe
        df=wage_reg[3]
        dat=df[:,:resid_mean]
        features=collect(Vector{Float64}(dat)')
        result = kmeans(features, 5; maxiter=100, display=:iter)
        a=assignments(result)
        centers=result.centers
        clusters=DataFrame(MID=df.MID,cluster=a)
    else
        df=wage_reg[2]
        dat=df[:,:resid_mean]
        features=collect(Vector{Float64}(dat)')
        result = kmeans(features, 5; maxiter=100, display=:iter)
        a=assignments(result)
        centers=result.centers
        clusters=DataFrame(MID=df.MID,cluster=a)
    end
    return clusters,centers
end
=======
vl=[ed_dummies;:age_mother]
df=df[3]
df=df[:,[vl;:resid_mean]]
features=collect(Matrix{Float64}(df)')
>>>>>>> b9ea32a4280baaa33fc1799564b8c26de3de6e7b


test=wage_clustering(df,true)

<<<<<<< HEAD
####testing
=======
a=assignments(result)
c = counts(result) # get the cluster sizes
M = R.centers
>>>>>>> b9ea32a4280baaa33fc1799564b8c26de3de6e7b

<<<<<<< HEAD
    wage_reg=df

    df=wage_reg[3]
    dat=df[:,:resid_mean]
    features=collect(Vector{Float64}(dat)')
    result = kmeans(features, 5; maxiter=100, display=:iter)
    a=assignments(result)
    centers=result.centers
    clusters=DataFrame(MID=df.MID,cluster=a)




####testing

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

=======
>>>>>>> b9ea32a4280baaa33fc1799564b8c26de3de6e7b