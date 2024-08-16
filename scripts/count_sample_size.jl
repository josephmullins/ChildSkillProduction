include("../src/model.jl")
include("../src/model_older_children.jl")
include("../src/estimation.jl")

# =======================   read in the data ===================================== #
# - load the data 
panel_data = DataFrame(CSV.File("../../../PSID_CDS/data-derived/psid_fam.csv",missingstring = ["","NA"]))
panel_data, m_ed, f_ed = prep_data(panel_data)
panel_data = DataFrame(filter(x-> sum(skipmissing(x.ind_not_sample.==0))>0 || sum(x.all_prices)>0,groupby(panel_data,:kid)))

Ik = panel_data.year.==1997 .|| panel_data.year.==2002 .|| panel_data.year.==2007
panel_data = panel_data[Ik,:]

function check_in_sample(d)
    in_sample = false
    for it in 1:3
        if (d.prices_observed[it] && (d.age[it]<=12) && (d.ind_not_sample[it]==0))
            in_sample = in_sample || (d.mtime_valid[it] & d.chcare_valid[it])
            if !ismissing.(d.log_good[it])
                in_sample = in_sample || d.mtime_valid[it] || d.chcare_valid[it] || d.ftime_valid[it]
            end
        end
        if it<3
            if d.all_prices[it] && d.mtime_valid[it] && d.mtime_valid[it+1] && (d.age[it]<=12)
                i = d.mtime_valid[it+1] && d.AP_valid[it] && d.AP_valid[it+1] && d.LW_valid[it]
                in_sample = in_sample || i
                i = d.mtime_valid[it+1] && d.AP_valid[it] && d.LW_valid[it+1] && d.LW_valid[it]
                in_sample = in_sample || i
            end
        end
    end
    return in_sample
end

d = filter(x->check_in_sample(x),groupby(panel_data,:kid))
N = length(d)

println("The sample size is $N")
