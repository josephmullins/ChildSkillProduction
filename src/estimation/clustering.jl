using Clustering
# - this script contains functions for clustering
# ------------------------------------------------------------------- #
# - this function runs the clustering routine given data in dat and given a list of variables to use.

function cluster_routine_robust(dat,vlist,nclusters,maxiter = 100)
    lW,X,d = get_wage_data(dat,[:constant;vlist],false)
    # get an initial assignment by clustering on residuals

    coef=inv(X'X)*X'lW
    d[!,:resid] = d.logwage_m .- predict_wage(d,[:constant;vlist],coef)
    dn = combine(groupby(d,:MID),:resid => mean)
    result = kmeans(dn[!,:resid_mean]', nclusters; maxiter=100, display=:iter)
    dn[!,:cluster] = result.assignments
    select!(dn,Not(:resid_mean))
    d = innerjoin(d,dn,on=:MID)

    eps = Inf
    iter = 0
    assignment = result.assignments
    gd = groupby(d,:MID)
    μk0 = zeros(nclusters)
    X = X[:,2:end] #<- drop the intercept term
    while (eps>1e-10) & (iter<maxiter)
        iter +=1 
        # calculate the group fixed effect
        μk,β = wagereg_group_fe(d,vlist)
        # calculate the residual error *not including* the fe:
        d[!,:xb] = d[!,:logwage_m] .- X*β
        # get a new assignment of individuals to groups
        #μk = 0.5*μk + 0.5*μk0
        assignment_new = [assign_cluster(d,μk) for d in gd]
        eps = sum((μk .- μk0).^2)
        #eps = sum((assignment .- assignment_new).^2)
        dn[!,:cluster] = assignment_new
        assignment[:] = assignment_new
        #println(μk)
        μk0[:] = μk
        select!(d,Not(:cluster))
        d = innerjoin(d,dn,on=:MID)
        println(eps," ",iter)
    end
    relabel = sortperm(μk0) #<- this tells the ordering of the cluster
    position_map = [findfirst(relabel.==k) for k=1:nclusters] #<- this uses the ordering to map each cluster to its new position
    dn[!,:cluster] = position_map[assignment]
    dn[!,:center] = μk0[assignment]
    return dn
end


# - Assigns dataframe d to a new cluster given the current parameters
function assign_cluster(d,μk)
    nclusters = length(μk)
    ssq = [sum((d.xb .- μk[k]).^2) for k in 1:nclusters]
    return argmin(ssq)
end

# - Runs a wage regression given the current cluster assignment to get new wage parameters
function wagereg_group_fe(dat,vlist)
    cluster_dummies=make_dummy(dat,:cluster)
    ncluster = length(cluster_dummies)
    lW,X,d = get_wage_data(dat,[cluster_dummies;vlist],false)
    coef=inv(X'X)*X'lW
    μk = coef[1:ncluster]
    β = coef[ncluster+1:end]
    return μk,β
end

# - Function to return the required data from vlist
function get_wage_data(data,vlist::Array{Symbol,1},fe)
    N = size(data)[1]
    d = dropmissing(select(data, [:MID;:logwage_m;vlist]))
    d2 = copy(d)
    # if including individual fixed effect, first de-mean here
    if fe
        demean(x) = x .- mean(x)
        gd = groupby(d2,:MID)
        for v in [:logwage_m;vlist]
            transform!(gd,v => demean => v)
        end
    end
    lW = Vector{Float64}(d2.logwage_m)
    return lW,Matrix{Float64}(d2[!,vlist]),d
end