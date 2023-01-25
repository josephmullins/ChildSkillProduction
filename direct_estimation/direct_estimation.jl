using Parameters, Random, Printf
# not sure where to put this right now, but will leave it here for the time being
# this script estimates the CES function directly assuming no endogeneity of inputs and inputs measured without error


# issue with this setup: we want the factor shares and TFP to be flexible parameters that can be updated
@with_kw struct CESmod{R}
    ρ::R = -1.5
    γ::R = -3.
    σ_η::Float64 = 0.5 #<- we don't need to take a derivative with respect to this parameter so we can fix the type.
    δ::Array{R,1} = [0.05,0.95]
    βm::Array{R,1} = zeros(2)
    βf::Array{R,1} = zeros(2)
    βg::Array{R,1} = zeros(2)
    βθ::Array{R,1} = zeros(2)
    βY::Array{R,1} = zeros(2)
end

# this function returns the parameters given a vector x and a specification spec which is a named dictionary
function CESmod(x,spec)
    δ = x[1:2]
    ρ = x[3]
    γ = x[4]
    nm,nf,ng,nθ,nY = (length(v) for v in spec) #<- this assumes we have the right ordering
    βm = x[5:5+nm]
    pos = 5+nm+1
    βf = x[pos:pos+nf]
    pos += nf+1
    βg = x[pos:pos+ng]
    pos += ng+1
    βθ = x[pos:pos+nθ]
    pos += nθ+1
    βY = [1;x[pos:pos+nY-2]]
    return CESmod(ρ = ρ,γ = γ,δ = δ,βm = βm,βf = βf,βg = βg,βθ = βθ,βY = βY)
end

function CESmod(spec)
    P = CESmod()
    nm,nf,ng,nθ = (length(v) for v in spec)
    return reconstruct(P,βm=zeros(nm+1),βf = zeros(nf+1),βg = zeros(ng+1),βθ = zeros(nθ+1))
end


function CESvec(P::CESmod)
    return [P.δ;P.ρ;P.γ;P.βm;P.βf;P.βg;P.βθ;P.βY[2:end]]
end

# GMM problem: have a residual function, and an instrument function?

# a simple utility function that returns a linear combination of β and each variable in the list vars
# -- this function assumes that all specifications require a constant
function linear_combination(β,vars,data,n)
    r = β[1] #<- assuming a constant term
    for j in eachindex(vars)
       r += β[j+1]*data[vars[j]][n]
    end
    return r
end

# function to get the relevant inputs based on marital status
function get_inputs(P,spec,data,n)
    Y = Y_input(P,spec,data,n)
    if data.mar[n]==1
        return data.mtime[n],data.ftime[n],data.goods[n],Y,data.LW97[n]
    else
        return data.mtime[n],data.goods[n],Y,data.LW97[n]
    end
end

# function takes data, observation (n), + model params and returns the factor shares
function factor_share(P::CESmod,spec,data,n)
    @unpack βm,βf,βg = P #<- so model would need to contain these parameters
    am = exp(linear_combination(βm,spec.vm,data,n))
    af = exp(linear_combination(βf,spec.vf,data,n))
    ag = exp(linear_combination(βg,spec.vg,data,n))
    return am,af,ag
end

function Y_input(P::CESmod,spec,data,n)
    Y = 0
    @unpack βY = P
    for j in eachindex(spec.vY) #<- iterate over variables that count in childcare input
        Y += exp(βY[j])*data[spec.vY[j]][n]
    end
    return Y
end

# function does the same as above, but returns logTFP
function logTFP(P::CESmod,spec,data,n)
    return linear_combination(P.βθ,spec.vθ,data,n)
end

# is this right? can coefficients offset each other in observables here??
function logCES(τ_m,τ_f,g,Y,logΨ_0,logθ,am,af,ag,ρ,γ,δ) #<- CES functions for married couples
    home_input = (am*τ_m^ρ + af*τ_f^ρ + ag*g^ρ)^(1/ρ)
    return logθ + (δ[1]/γ)*log(home_input^γ + Y^γ) + δ[2]*logΨ_0
end
function logCES(τ_m,g,Y,logΨ_0,logθ,am,ag,ρ,γ,δ) #<- CES functions for single mothers
    home_input = (am*τ_m^ρ + ag*g^ρ)^(1/ρ)
    return logθ + (δ[1]/γ)*log(home_input^γ + Y^γ) + δ[2]*logΨ_0
end


# -- function to predict skills given parameters and inputs
function predict_skill(P::CESmod,spec,data,n)
    @unpack ρ,γ,δ = P
    am,af,ag = factor_share(P,spec,data,n)
    logθ = logTFP(P,spec,data,n)
    inputs = get_inputs(P,spec,data,n)
    if data.mar[n]==0
        return logCES(inputs...,logθ,am,ag,ρ,γ,δ)
    else
        return logCES(inputs...,logθ,am,af,ag,ρ,γ,δ)
    end
end

# -- given instruments, Z, this is the function for GMM estimation
function gfunc(P::CESmod,Z,spec,data,n)
    resid = data.LW02[n] - predict_skill(P,spec,data,n)
    @views return resid*Z[:,n]
end

# -- GMM criterion
function gfunc(x,Z,spec,data,index)
    N = length(data.LW02)
    P = CESmod(x,spec)
    nm = size(Z)[1]
    g0 = zeros(nm)
    for n in index
        g0 += gfunc(P,Z,spec,data,n)
    end
    return g0/N
end

# -- NLLS criterion
function sumsq(x,spec,data)
    N = length(data.LW02)
    P = CESmod(x,spec)
    ssq = 0
    for n=1:N
        ssq += (data.LW02[n] - predict_skill(P,spec,data,n))^2
    end
    return ssq
end

# -- function to calculate the instruments given an initial guess
function get_instruments(x0,up,spec,data)
    nz = length(x0)
    N = length(data.LW02)
    Z = zeros(nz,N)
    for n in 1:N
        f(x) = predict_skill(CESmod(up(x),spec),spec,data,n)
        Z[:,n] = ForwardDiff.gradient(f,x0)
    end
    # BUT, we don't want to use LW97 as an instrument for itself, so we have to replace that line
    Z[2,:] = data.AP97
    return Z
end

function gmm_criterion(x,Z,spec,data,index=1:length(data.LW02))
    gn = gfunc(x,Z,spec,data,index)
    return sum(gn.^2)
end

function moment_variance(x,Z,spec,data)
    N = length(data.LW02)
    P = CESmod(x,spec)
    nm = size(Z)[1]
    G0 = zeros(nm,nm)
    for n=1:N
        gn = gfunc(P,Z,spec,data,n)
        G0 += gn*gn'
    end
    return G0/N
end

function parameter_variance(x_est,Z,spec,data,up)
    N = length(data.LW02)
    dg = ForwardDiff.jacobian(x->gfunc(up(x),Z,spec,data,1:N),x_est)
    V = moment_variance(up(x_est),Z,spec,data)
    dg_inv = inv(dg)
    return dg_inv*V*dg_inv'/N
end

function bootstrap(x_est,Z,spec,data,seed=1234,B=50,verbose=true)
    Random.seed!(seed)
    X = zeros(length(x_est),B)
    N = length(data.LW02)
    for b in 1:B
        if verbose
            println(" -- doing bootstrap trial $b of $B --")
        end
        ib = rand(1:N,N)
        #res = optimize(x->gmm_criterion(x,Z,spec,data,ib),x_est,LBFGS(),autodiff = :forward)
        res = optimize(x->gmm_criterion(x,Z,spec,data,ib),x_est,NelderMead())
        X[:,b] = res.minimizer
    end
    return X
end

# fix this later to integrate with function above (update = x->x) is default argument or similar
function bootstrap(x_est,Z,spec,data,update,seed=1234,B=50,verbose=true)
    Random.seed!(seed)
    X = zeros(length(x_est),B)
    N = length(data.LW02)
    for b in 1:B
        if verbose
            println(" -- doing bootstrap trial $b of $B --")
        end
        ib = rand(1:N,N)
        #res = optimize(x->gmm_criterion(update(x),Z,spec,data,ib),x_est,NelderMead())
        res = optimize(x->gmm_criterion(update(x),Z,spec,data,ib),x_est,LBFGS(),autodiff=:forward)
        X[:,b] = res.minimizer
    end
    return X
end


# --- Output functions (?)

function write_line!(io,format,M,v::Symbol,i::Int=0,vname::String="")
    nspec = length(M)
    write(io,vname,"&")
    for s in 1:nspec
        if i==0
            write(io,format(getfield(M[s],v)),"&")
        else
            write(io,format(getfield(M[s],v)[i]),"&")
        end
    end
    write(io,"\\\\","\n")
end

function write_observables!(io,format,formatse,M,SE,specs,labels,var::Symbol,specvar::Symbol,constant=true)

    nspec = length(M)
    if constant
        # write the constant:
        write_line!(io,format,M,var,1,"Const.")
        write_line!(io,formatse,SE,var,1)
    end
    vlist = union([s[specvar] for s in specs]...)
    for v in vlist
        vname = labels[v] #<-?
        write(io,vname,"&")
        # write estimates
        for j in 1:nspec
            i = findfirst(specs[j][specvar].==v)
            if isnothing(i)
                write(io,"-","&")
            else
                if constant
                    write(io,format(getfield(M[j],var)[1+i]),"&")
                else
                    write(io,format(getfield(M[j],var)[i]),"&")
                end
            end
        end
        write(io,"\\\\")
        # write standard errors
        write(io,"&")
        for j in 1:nspec
            i = findfirst(specs[j][specvar].==v)
            if isnothing(i)
                write(io,"","&")
            else
                if constant
                    write(io,formatse(getfield(SE[j],var)[1+i]),"&")
                else
                    write(io,formatse(getfield(SE[j],var)[i]),"&")
                end
            end
        end
        write(io,"\\\\")

    end
end

# specs are an array
function writetable(M,SE,specs,labels,outfile::String)
    form(x) = @sprintf("%0.2f",x)
    formse(x) = string("(",@sprintf("%0.2f",x),")")
    
    nspec = length(M)
    
    # Write the header
    io = open(outfile, "w");
    write(io,"\\begin{tabular}{l",repeat("c",nspec+1),"}\\\\\\toprule","\n")
    write(io,"&",["($s)&" for s in 1:nspec]...,"\\\\\\cmidrule(r){2-$(nspec+2)}")

    
    # Write the elasticity parameters 
    v = [:ρ,:γ]
    vname = ["\$\\rho\$","\$\\gamma\$"]
    for j in 1:2
        write_line!(io,form,M,v[j],0,vname[j])
        write_line!(io,formse,SE,v[j],0)
    end
    # δ_1 and δ_2
    for j in 1:2
        write_line!(io,form,M,:δ,j,"\$\\delta_{$j}\$")
        write_line!(io,formse,SE,:δ,j)
    end

    # Factor share Parameters:
    # a_{m}
    # write the header:
    write(io,"& \\multicolumn{$(nspec+1)}{c}{\$\\phi_{m}\$: Mother's Time}\\\\\\cmidrule(r){2-$(nspec+2)}")
    write_observables!(io,form,formse,M,SE,specs,labels,:βm,:vm)
    # a_{f}
    write(io,"& \\multicolumn{$(nspec+1)}{c}{\$\\phi_{f}\$: Father's Time}\\\\\\cmidrule(r){2-$(nspec+2)}")
    write_observables!(io,form,formse,M,SE,specs,labels,:βf,:vf)
    # a_{g}
    write(io,"& \\multicolumn{$(nspec+1)}{c}{\$\\phi_{g}\$: Goods}\\\\\\cmidrule(r){2-$(nspec+2)}")
    write_observables!(io,form,formse,M,SE,specs,labels,:βg,:vg)

    # a_{Y}
    write(io,"& \\multicolumn{$(nspec+1)}{c}{Childcare}\\\\\\cmidrule(r){2-$(nspec+2)}")
    write_observables!(io,form,formse,M,SE,specs,labels,:βY,:vY,false)


    # Write the TFP parameters
    write(io,"& \\multicolumn{$(nspec+1)}{c}{\$\\phi_{\\theta}\$: TFP}\\\\\\cmidrule(r){2-$(nspec+2)}")
    write_observables!(io,form,formse,M,SE,specs,labels,:βθ,:vθ)

    # Write Other Summary Statistics?



    write(io,"\\bottomrule")
    write(io,"\\end{tabular}")
    close(io)

end


# par = update(res[2],1.,1.)
# se = update(se,1.,1.)

# #write(io,"\\multicolumn{4}{c}{Elasticities} & & & & &\\\\")
# write(io,"\$\\rho\$ & \$\\gamma\$ & \$\\delta_1\$ & \$\\delta_2\$ & & \$\\lambda\$ & & \\\\")
# write(io,string(form(par.ρ)," & ",form(par.γ)," & ",form(par.δ[1])," & ",form(par.δ[2]),"  & & ",form(par.λ)," & & \\\\  "))
# write(io,string(formse(se.ρ)," & ",formse(se.γ)," & ",formse(se.δ[1])," & ",formse(se.δ[2]),"  & &", formse(se.λ)," & & \\\\ \\noalign{\\medskip}  "))
# write(io,"\\multicolumn{8}{c}{\$\\phi_g\$: Market Goods} \\\\\\cmidrule(r){1-8}")
# write(io,obs_headers)
# write(io,string(form(par.βg[1])," &",form(par.βg[2])," & ",form(par.βg[3])," & ",form(par.βg[4])," & ",form(par.βg[5])," & ",form(par.βg[6])," & ",form(par.βg[7])," & ",form(par.βg[8]),"\\\\"))
# write(io,string(formse(se.βg[1])," &",formse(se.βg[2])," & ",formse(se.βg[3])," & ",formse(se.βg[4])," & ",formse(se.βg[5])," & ",formse(se.βg[6])," & ",formse(se.βg[7])," & ",formse(se.βg[8]),"\\\\ \\noalign{\\medskip} "))
# write(io,"\\multicolumn{8}{c}{\$\\phi_m\$: Mother's Time} \\\\\\cmidrule(r){1-8}")
# write(io,obs_headers)
# write(io,string(form(par.βm[1])," &",form(par.βm[2])," & ",form(par.βm[3])," &",form(par.βm[4])," & & & ",form(par.βm[5])," & ",form(par.βm[6]),"\\\\"))
# write(io,string(formse(se.βm[1])," &",formse(se.βm[2])," & ",formse(se.βm[3])," &",formse(se.βm[4])," & & & ",formse(se.βm[5])," & ",formse(se.βm[6]),"\\\\ \\noalign{\\medskip}  "))
# write(io,"\\multicolumn{8}{c}{\$\\phi_f\$: Father's Time} \\\\\\cmidrule(r){1-8}")
# write(io,obs_headers)
# write(io,string(form(par.βf[1]),"& & & & ",form(par.βf[2])," &",form(par.βf[3])," & ",form(par.βf[4])," & ",form(par.βf[5]),"\\\\"))
# write(io,string(formse(se.βf[1]),"& & & & ",formse(se.βf[2])," &",formse(se.βf[3])," & ",formse(se.βf[4])," & ",formse(se.βf[5]),"\\\\"))
# write(io,"\\multicolumn{8}{c}{\$\\phi_\\theta\$: TFP} \\\\\\cmidrule(r){1-8}")
# write(io,theta_headers)
# write(io,string(form(par.βθ[1])," &",form(par.βθ[3])," & ",form(par.βθ[4])," & ",form(par.βθ[5])," & ",form(par.βθ[6])," & ",form(par.βθ[7])," & ",form(par.βθ[2])," & \\\\"))
# write(io,string(formse(se.βθ[1])," &",formse(se.βθ[3])," & ",formse(se.βθ[4])," & ",formse(se.βθ[5])," & ",formse(se.βθ[6])," & ",formse(se.βθ[7])," & ",formse(se.βθ[2])," & \\\\ \\noalign{\\medskip} "))
# close(io)

