
struct child_data
    Xm::Matrix{Float64}
    Xf::Matrix{Float64}
    Xy::Matrix{Float64}
    Xθ::Matrix{Float64}
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

function child_data(data,spec)
    Xm = hcat([data[!,v] for v in spec.vm]...)'
    Xf = hcat([data[!,v] for v in spec.vf]...)'
    Xy = hcat([data[!,v] for v in spec.vy]...)'
    Xθ = hcat([data[!,v] for v in spec.vθ]...)'
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
    ismissing.(data.AP),
    ismissing.(data.LW)
    )
end
