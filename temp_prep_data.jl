
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
panel_data[!,:AP] = coalesce.(panel_data.AP,0.)
panel_data[!,:LW] = coalesce.(panel_data.LW,0.)
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
i97 = panel_data.year.==1997
i02 = panel_data.year.==2002
i07 = panel_data.year.==2007
ii = panel_data.all_prices .& panel_data.mtime_valid .& (panel_data.age.<=12)
panel_data.LW[i97] .-= mean(panel_data.LW[i97 .& ii])
panel_data.AP[i97] .-= mean(panel_data.AP[i97 .& ii])
panel_data.LW[i02] .-= mean(panel_data.LW[i02 .& panel_data.all_prices .& panel_data.mtime_valid])
panel_data.AP[i02] .-= mean(panel_data.AP[i02 .& panel_data.all_prices .& panel_data.mtime_valid])
panel_data.LW[i07] .-= mean(panel_data.LW[i07 .& panel_data.all_prices .& panel_data.mtime_valid])
panel_data.AP[i07] .-= mean(panel_data.AP[i07 .& panel_data.all_prices .& panel_data.mtime_valid])

#vlist = [v_demogs;v_prices;v_inputs]
#data = NamedTuple(zip((v for v in vlist),(panel_data[!,v] for v in vlist)))

# other variables
# :m_pc97
# :m_race 
# :f_race 
# :curr_married 
# :m_white  
# :m_black
# :m_r_oth 
# :fed_hsd
# :fed_hs 
# :fed_scoll 
# :fed_coll 
# :fed_postcol
# :fed_collplus 
# :fed_scollplus 
# :med_hsd 
# :med_scoll 
# :med_coll 
# :med_postcol
# :med_collplus
# :med_scollplus 
