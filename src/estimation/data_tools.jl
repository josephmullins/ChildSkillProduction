
# a struct that pre-determines the data we will use in estimation. not strictly necessary but keeps me organized.
struct child_data
    Xm::Matrix{Float64}
    Xf::Matrix{Float64}
    Xy::Matrix{Float64}
    Xθ::Matrix{Float64}
    Zc::Matrix{Float64}
    Zf::Matrix{Float64}
    Zg::Matrix{Float64}
    Zτ::Matrix{Float64}
    Ω::Matrix{Float64}
    year::Vector{Int64}
    age::Vector{Int64}
    # logicals:
    mar_stat::Vector{Bool}
    prices_observed::Vector{Bool}
    ind_not_sample::Vector{Bool}
    # prices 
    logprice_g::Vector{Float64}
    logprice_c::Vector{Float64}
    logwage_m::Vector{Float64}
    logwage_f::Vector{Float64}
    log_total_income::Vector{Float64}
    # inputs
    log_mtime::Vector{Float64}
    log_ftime::Vector{Float64}
    log_chcare::Vector{Float64}
    log_good::Vector{Float64}

    mtime_missing::Vector{Bool}
    ftime_missing::Vector{Bool}
    chcare_missing::Vector{Bool}
    goods_missing::Vector{Bool}

    # instruments:
    Z::Vector{Matrix{Float64}}

    # skills:
    all_prices::Vector{Bool} #<- indicates whether the observation can be used for skill outcome moments
    AP::Vector{Float64}
    LW::Vector{Float64}
    AP_missing::Vector{Bool}
    LW_missing::Vector{Bool}

end

# function to create the struct above.
function child_data(data,spec)
    Xm = hcat([data[!,v] for v in spec.vm]...)'
    Xf = hcat([data[!,v] for v in spec.vf]...)'
    Xy = hcat([data[!,v] for v in spec.vy]...)'
    Xθ = hcat([data[!,v] for v in spec.vθ]...)'
    if :zc in keys(spec)
        Zτ = hcat([data[!,v] for v in spec.zτ]...)'
        Ω = hcat([data[!,v] for v in spec.vΩ]...)'
        Zc = hcat([data[!,v] for v in spec.zc]...)'
        Zf = hcat([data[!,v] for v in spec.zf]...)'
        Zg = hcat([data[!,v] for v in spec.zg]...)'
    else
        Zτ = zeros(0,0)
        Ω = zeros(0,0)
        Zc = zeros(0,0)
        Zf = zeros(0,0)
        Zg = zeros(0,0)
    end
    Z = []
    for zv in spec.zlist_97
        push!(Z,coalesce.(hcat([data[data.year.==1997,v] for v in zv]...)',0.))
    end
    for zv in spec.zlist_02
        push!(Z,coalesce.(hcat([data[data.year.==2002,v] for v in zv]...)',0.))
    end
    for zv in spec.zlist_07
        push!(Z,coalesce.(hcat([data[data.year.==2007,v] for v in zv]...)',0.))
    end
    # have to update the specification
    for zv in spec.zlist_prod
        Zt = [] #
        for i in eachindex(zv)
            y = 1997+spec.zlist_prod_t[i]
            for v in zv[i]
                push!(Zt,Vector{Float64}(coalesce.(data[data.year.==y,v],0.)))
            end
        end
        push!(Z,hcat(Zt...)')
    end
    #push!(Z,[Matrix{Float64}(undef,1,0) for i in 1:8]...)
    for zv in spec.zlist_prod
        Zt = []
        for i in eachindex(zv)
            y = 2002+spec.zlist_prod_t[i]
            for v in zv[i]
                push!(Zt,Vector{Float64}(coalesce.(data[data.year.==y,v],0.)))
            end
        end
        push!(Z,hcat(Zt...)')
    end
    return child_data(coalesce.(Xm,0.),
    coalesce.(Xf,0.),
    coalesce.(Xy,0.),
    coalesce.(Xθ,0.),
    coalesce.(Zc,0.),
    coalesce.(Zf,0.),
    coalesce.(Zg,0.),
    coalesce.(Zτ,0.),
    coalesce.(Ω,0.),
    data.year,
    data.age,
    coalesce.(data.mar_stat,false),
    data.prices_observed,
    data.ind_not_sample,
    coalesce.(data.logprice_g,0.),
    coalesce.(data.logprice_c,0.),
    coalesce.(data.logwage_m,0.),
    coalesce.(data.logwage_f,0.),
    coalesce.(data.log_total_income,0.),
    coalesce.(data.log_mtime,0.),
    coalesce.(data.log_ftime,0.),
    coalesce.(data.log_chcare,0.),
    coalesce.(data.log_good,0.),
    ismissing.(data.log_mtime),
    ismissing.(data.log_ftime),
    ismissing.(data.log_chcare),
    ismissing.(data.log_good),
    Z,
    panel_data.all_prices,
    coalesce.(data.AP,0.),
    coalesce.(data.LW,0.),
    #ismissing.(data.AP), ERROR HERE
    #ismissing.(data.LW), ERROR HERE
    .!data.AP_valid,
    .!data.LW_valid
    )
end

# a function to clean the data frame that we're using. Combine with the function above?
function prep_data(panel_data)
    # demographics
    #v_demogs = [:year;:mar_stat;:div;:constant;m_ed;f_ed;:num_0_5;:age;cluster_dummies[2:end];:mu_k]

    panel_data.m_ed = replace(panel_data.m_ed,">16" => "16","<12" => "12") #<-  simplify the education categories
    panel_data.f_ed = replace(panel_data.f_ed,">16" => "16","<12" => "12")
    m_ed = make_dummy(panel_data,:m_ed)
    f_ed = make_dummy(panel_data,:f_ed)
    panel_data[!,:constant] .= 1.
    panel_data[!,:mar_stat] = panel_data.curr_married.==1
    panel_data[!,:div] = panel_data.curr_married.==0
    panel_data[!,f_ed] = coalesce.(panel_data[:,f_ed],0.)
    panel_data[!,:all_prices] = panel_data.ind_price_97_01.==1
    i02 = panel_data.year.>=2002
    panel_data[i02,:all_prices] = panel_data.ind_price_02_06[i02].==1
    panel_data[!,:ind02] .= panel_data.year.==2002
    panel_data[!,:agesq] .= panel_data.age.^2

    # prices:
    #v_prices = [:prices_observed,:logwage_m,:logwage_f,:logprice_g,:logprice_c,:logprice_c_m,:logprice_m_f,:logprice_c_g,:logprice_m_g,:logprice_f_g]

    panel_data[!,:logwage_m] = panel_data.ln_wage_m
    panel_data[!,:logwage_f] = coalesce.(panel_data.ln_wage_f,0) #<- make these into zeros to avoid a problem with instruments
    panel_data[!,:prices_observed] = panel_data.ind_price.==1
    panel_data[!,:logprice_g] = log.(panel_data.p_avg)
    panel_data[!,:logprice_c] = log.(panel_data.p_yocent_e_cps_cpkt) .- log(33*52)
    ### relative prices
    panel_data[!,:logprice_c_m] = panel_data.logprice_c .- panel_data.logwage_m
    panel_data[!,:logprice_m_f] = panel_data.logwage_m .- panel_data.logwage_f
    panel_data[!,:logprice_c_g] = panel_data.logprice_c .- panel_data.logprice_g
    panel_data[!,:logprice_m_g] = panel_data.logwage_m .- panel_data.logprice_g
    panel_data[!,:logprice_f_g] = panel_data.logwage_f .- panel_data.logprice_g


    # inputs and outputs
    #v_inputs = [:log_mtime,:log_ftime,:log_chcare,:log_good,:log_total_income,:AP,:LW]

    panel_data[!,:log_mtime] = panel_data.ln_tau_m
    panel_data[!,:log_ftime] = panel_data.ln_tau_f
    panel_data[!,:log_chcare] = panel_data.ln_chcare_exp
    panel_data[!,:log_good] = panel_data.ln_hhinvest
    #panel_data[!,:log_total_income] = log.(panel_data.m_wage .+ coalesce.(panel_data.f_wage,0))
    panel_data[!,:log_total_income] = log.(exp.(panel_data.logwage_m) .+ exp.(panel_data.logwage_f))
    panel_data[!,:AP_valid] = .!ismissing.(panel_data.AP)
    panel_data[!,:LW_valid] = .!ismissing.(panel_data.LW)
    #panel_data[!,:AP] = coalesce.(panel_data.AP,0.)
    #panel_data[!,:LW] = coalesce.(panel_data.LW,0.)
    panel_data[!,:mtime_valid] = .!ismissing.(panel_data.log_mtime) .& .!ismissing.(panel_data.m_ed)
    panel_data[!,:ftime_valid] = .!ismissing.(panel_data.log_ftime) #.| .!panel_data.mar_stat only use non-missing father's time

    panel_data[!,:chcare_valid] = .!ismissing.(panel_data.log_chcare)

    # these two lines cause problems with estimation. 
    # why? because missing(data.log_mtime[it]) (for example) is called calc_demand_resids!. If we set to zero, no longer coded as missing
    panel_data[!,:log_mtime_coalesced] = coalesce.(panel_data.log_mtime,0.)
    panel_data[!,:log_ftime_coalesced] = coalesce.(panel_data.log_ftime,0.)
    panel_data[!,:log_chcare_input] = coalesce.(panel_data.log_chcare .- panel_data.logprice_c,0.)
    panel_data[!,:log_good_input] = coalesce.(panel_data.log_good .- panel_data.logprice_g,0.)

    # de-mean test scores in all years
    # i97 = panel_data.year.==1997
    # i02 = panel_data.year.==2002
    # i07 = panel_data.year.==2007
    # ii = panel_data.all_prices .& panel_data.mtime_valid .& (panel_data.age.<=12)
    # panel_data.LW[i97] .-= mean(panel_data.LW[i97 .& ii])
    # panel_data.AP[i97] .-= mean(panel_data.AP[i97 .& ii])
    # panel_data.LW[i02] .-= mean(panel_data.LW[i02 .& panel_data.all_prices .& panel_data.mtime_valid])
    # panel_data.AP[i02] .-= mean(panel_data.AP[i02 .& panel_data.all_prices .& panel_data.mtime_valid])
    # panel_data.LW[i07] .-= mean(panel_data.LW[i07 .& panel_data.all_prices .& panel_data.mtime_valid])
    # panel_data.AP[i07] .-= mean(panel_data.AP[i07 .& panel_data.all_prices .& panel_data.mtime_valid])
    return panel_data, m_ed, f_ed
end

# function to make a set of dummy variables in the dataframe
function make_dummy(data,var::Symbol)
    vals = sort(unique(skipmissing(data[!,var])))
    nvals = length(vals)
    vnames = [Symbol(var,"_",x) for x in vals]
    for i in 1:nvals
        data[!,vnames[i]] = data[!,var].==vals[i]
    end
    return vnames
end

# function to create interaction terms in the data and spit out a list of their names
function make_interactions(data,V1::Vector{Symbol},V2::Vector{Symbol})
    names = []
    for v1 in V1, v2 in V2
        name = Symbol(v1,"_x_",v2)
        data[!,name] = data[!,v1].*data[!,v2]
        push!(names,name)
    end
    return names
end

# function get_data_method_1(data,spec)
#     Xm = hcat([data[!,v] for v in spec.vm]...)'
#     Xf = hcat([data[!,v] for v in spec.vf]...)'
#     Xy = hcat([data[!,v] for v in spec.vy]...)'
#     Xθ = hcat([data[!,v] for v in spec.vθ]...)'
#     X = hcat([data[!,v] for v in spec.vx]...)'
#     Xτ = hcat([data[!,v] for v in spec.vx]...)'
#     Ω = hcat([data[!,v] for v in spec.vΩ]...)'
#     Z = []
#     for zv in spec.zlist_97
#         push!(Z,coalesce.(hcat([data[data.year.==1997,v] for v in zv]...)',0.))
#     end
#     for zv in spec.zlist_02
#         push!(Z,coalesce.(hcat([data[data.year.==2002,v] for v in zv]...)',0.))
#     end
#     for zv in spec.zlist_07
#         push!(Z,coalesce.(hcat([data[data.year.==2007,v] for v in zv]...)',0.))
#     end
#     # have to update the specification
#     for zv in spec.zlist_prod
#         Zt = [] #
#         for i in eachindex(zv)
#             y = 1997+spec.zlist_prod_t[i]
#             for v in zv[i]
#                 push!(Zt,Vector{Float64}(coalesce.(data[data.year.==y,v],0.)))
#             end
#         end
#         push!(Z,hcat(Zt...)')
#     end
#     #push!(Z,[Matrix{Float64}(undef,1,0) for i in 1:8]...)
#     for zv in spec.zlist_prod
#         Zt = []
#         for i in eachindex(zv)
#             y = 2002+spec.zlist_prod_t[i]
#             for v in zv[i]
#                 push!(Zt,Vector{Float64}(coalesce.(data[data.year.==y,v],0.)))
#             end
#         end
#         push!(Z,hcat(Zt...)')
#     end
#     return (;
#     Xm = coalesce.(Xm,0.),
#     Xf = coalesce.(Xf,0.),
#     Xy = coalesce.(Xy,0.),
#     Xθ = coalesce.(Xθ,0.),
#     year = data.year,
#     age = data.age,
#     mar_stat = coalesce.(data.mar_stat,false),
#     prices_observed = data.prices_observed,
#     data.ind_not_sample,
#     coalesce.(data.logprice_g,0.),
#     coalesce.(data.logprice_c,0.),
#     coalesce.(data.logwage_m,0.),
#     coalesce.(data.logwage_f,0.),
#     coalesce.(data.log_total_income,0.),
#     coalesce.(data.log_mtime,0.),
#     coalesce.(data.log_ftime,0.),
#     coalesce.(data.log_chcare,0.),
#     coalesce.(data.log_good,0.),
#     ismissing.(data.log_mtime),
#     ismissing.(data.log_ftime),
#     ismissing.(data.log_chcare),
#     ismissing.(data.log_good),
#     Z,
#     panel_data.all_prices,
#     coalesce.(data.AP,0.),
#     coalesce.(data.LW,0.),
#     ismissing.(data.AP),
#     ismissing.(data.LW)
#     )
# end