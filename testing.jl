# let's test whether stuff gets slowed down by data frames.

# don't need to keep this but it confirms that we will get a decent speedup if we convert the dataframe to a namedtuple with the data we want

using CSV, DataFrames
d = DataFrame(CSV.File("/Users/joseph/Dropbox/PSID_CDS/data-derived/psid_fam.csv"))

log_mtime = convert(Vector{Union{Float64,Missing}},d.ln_tau_m)

function test1(d,x)
    s = 0.
    for i in eachindex(d.ln_tau_m)
        if !ismissing(d.ln_tau_m[i])
            s += x*d.ln_tau_m[i]
        end
    end
end

function test2(lntau,x)
    s = 0.
    for i in eachindex(lntau)
        if !ismissing(lntau[i])
            s += x*lntau[i]
        end
    end
end

test1(d,0.1)
test2(log_mtime,0.1)

d2 = (a = 1,ln_tau_m = d.ln_tau_m)

@time test1(d,0.1)
@time test2(log_mtime,0.1)
@time test2(d.ln_tau_m,0.1)