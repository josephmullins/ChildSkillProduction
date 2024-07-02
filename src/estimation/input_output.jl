function pval_ind_detailed(p,char = "+")
    if p<0.001
        return string("^{",repeat(char,3),"}")
    elseif p<0.01
        return string("^{",repeat(char,2),"}")
    elseif p<0.05
        return string("^{",char,"}")
    else
        return ""
    end
end
function pval_ind(p,char = "+")
    if p<0.05
        return string("^{",char,"}")
    else
        return ""
    end
end
function format_pval(form,x,p)
    return string("\$",form(x),pval_ind(p),"\$")
end

# this function writes a table that summarizes the results of the restricted estimation from many specifications for the same case ("uc" or "nbs")
function write_joint_gmm_table_production(results,specs,labels,outfile::String)
    M = [update(results[r].est,specs[r],results[r].case) for r in eachindex(results)]
    SE = [update(results[r].se,specs[r],results[r].case) for r in eachindex(results)]
    Pp = [update_demand(results[r].p_indiv,specs[r]) for r in eachindex(results)]

    form(x) = @sprintf("%0.2f",x)
    formse(x) = string("(",@sprintf("%0.2f",x),")")
    nspec = length(M)
    
    midrule(s) = "\\cmidrule(r){$(2+(s-1)*nspec)-$(1+s*nspec)}"
    # Write the header:
    io = open(outfile, "w");
    write(io,"\\begin{tabular}{l",repeat("c",nspec*4),"}\\\\\\toprule","\n")

    # - work on elasticity parameters
    write(io," & \\multicolumn{$nspec}{c}{\$\\epsilon_{\\tau,g}\$} & \\multicolumn{$nspec}{c}{\$\\epsilon_{x,H}\$} & \\multicolumn{$nspec}{c}{\$\\delta_{1}\$} & \\multicolumn{$nspec}{c}{\$\\delta_{2}\$} ","\\\\\n")
    write(io,repeat("& \$(\\kappa=0)\$ & \$(\\kappa=1)\$ ",4)...,"\\\\",[midrule(s) for s in 1:4]...,"\n")
    
    # -- now write the estimates:
    
    e_ρ = [1/(1-M[s].ρ) for s in 1:nspec]
    se_ρ = [SE[s].ρ/(1-M[s].ρ)^2 for s in 1:nspec]
    e_γ = [1/(1-M[s].γ) for s in 1:nspec]
    se_γ = [SE[s].γ/(1-M[s].γ)^2 for s in 1:nspec]

    write(io,[string("&",format_pval(form,e_ρ[s],Pp[s].ρ)) for s in 1:nspec]...)
    write(io,[string("&",format_pval(form,e_γ[s],Pp[s].γ)) for s in 1:nspec]...)
    write(io,[string("&",form(M[s].δ[1])) for s in 1:nspec]...)
    write(io,[string("&",form(M[s].δ[2])) for s in 1:nspec]...) #automate this?
    write(io,"\\\\\n")
    write(io,[string("&",formse(se_ρ[s])) for s in 1:nspec]...)
    write(io,[string("&",formse(se_γ[s])) for s in 1:nspec]...)
    write(io,[string("&",formse(SE[s].δ[1])) for s in 1:nspec]...)
    write(io,[string("&",formse(SE[s].δ[2])) for s in 1:nspec]...) #automate this?
    write(io,"\\\\\n")
    write(io,repeat("&",4*nspec),"\\\\\n")

    # - Write factor share parameters
    write(io," & \\multicolumn{$nspec}{c}{\$\\tilde{\\phi}_{m}\$: Mother's Time} & \\multicolumn{$nspec}{c}{\$\\tilde{\\phi}_{f}\$: Father's Time} & \\multicolumn{$nspec}{c}{\$\\tilde{\\phi}_{x}\$: Childcare} & \\multicolumn{$nspec}{c}{\$\\phi_{\\theta}\$: TFP} ","\\\\\n")
    write(io,repeat("& \$(\\kappa=0)\$ & \$(\\kappa=1)\$ ",4)...,"\\\\",[midrule(s) for s in 1:4]...,"\n")

    vlist = union([s[specvar] for s in specs, specvar in [:vm,:vf,:vm,:vθ]]...)
    for v in vlist
        if v in keys(labels)
            vname = labels[v]
            vname = string(vname) #<-?
        else
            vname = string(v)
        end
        write(io,vname)
        # write estimates
        varlist = [:βm,:βf,:βy,:βθ]
        svarlist = [:vm,:vf,:vy,:vθ]#<- 
        for k in 1:4
            var = varlist[k]
            specvar = svarlist[k]
            for j in 1:nspec
                i = findfirst(specs[j][specvar].==v)
                if isnothing(i)
                    write(io,"&","-")
                else
                    xval = getfield(M[j],var)[i]
                    if var==:βθ
                        write(io,"&",form(xval))
                    else                        
                        pval = getfield(Pp[j],var)[i]
                        write(io,"&",format_pval(form,xval,pval))
                    end
                end
            end
        end
        write(io,"\\\\\n")
        # now write standard errors:
        
        for k in 1:4
            var = varlist[k]
            specvar = svarlist[k]
            for j in 1:nspec
                i = findfirst(specs[j][specvar].==v)
                if isnothing(i)
                    write(io,"&","-")
                else
                    write(io,"&",formse(getfield(SE[j],var)[i]))
                end
            end
        end
        write(io,"\\\\\n")
    end 
    write(io,"\\\\\n")
    write(io,"\\bottomrule")
    write(io,"\\end{tabular}")
    close(io)
end

# this function writes a table that summarizes the results of the restricted estimation from many specifications for the same case ("uc" or "nbs")
function write_production_table(results,specs,labels,outfile::String)
    M = [update(results[r].est,specs[r],results[r].case) for r in eachindex(results)]
    SE = [update(results[r].se,specs[r],results[r].case) for r in eachindex(results)]
    Pp = [update_demand(results[r].p_indiv,specs[r]) for r in eachindex(results)]

    form(x) = @sprintf("%0.2f",x)
    formse(x) = string("(",@sprintf("%0.2f",x),")")
    nspec = length(M)
    
    midrule(s) = "\\cmidrule(r){$(2+(s-1)*nspec)-$(1+s*nspec)}"
    # Write the header:
    io = open(outfile, "w");
    write(io,"\\begin{tabular}{l",repeat("c",nspec*4),"}\\\\\\toprule","\n")

    # - work on elasticity parameters
    write(io," & \\multicolumn{$nspec}{c}{\$\\epsilon_{\\tau,g}\$} & \\multicolumn{$nspec}{c}{\$\\epsilon_{x,H}\$} & \\multicolumn{$nspec}{c}{\$\\delta_{1}\$} & \\multicolumn{$nspec}{c}{\$\\delta_{2}\$} ","\\\\\n")
    write(io,repeat(["&($s)" for s in 1:nspec],4)...,"\\\\",[midrule(s) for s in 1:4]...,"\n")
    
    # -- now write the estimates:
    
    e_ρ = [1/(1-M[s].ρ) for s in 1:nspec]
    se_ρ = [SE[s].ρ/(1-M[s].ρ)^2 for s in 1:nspec]
    e_γ = [1/(1-M[s].γ) for s in 1:nspec]
    se_γ = [SE[s].γ/(1-M[s].γ)^2 for s in 1:nspec]

    write(io,[string("&",format_pval(form,e_ρ[s],Pp[s].ρ)) for s in 1:nspec]...)
    write(io,[string("&",format_pval(form,e_γ[s],Pp[s].γ)) for s in 1:nspec]...)
    write(io,[string("&",form(M[s].δ[1])) for s in 1:nspec]...)
    write(io,[string("&",form(M[s].δ[2])) for s in 1:nspec]...) #automate this?
    write(io,"\\\\\n")
    write(io,[string("&",formse(se_ρ[s])) for s in 1:nspec]...)
    write(io,[string("&",formse(se_γ[s])) for s in 1:nspec]...)
    write(io,[string("&",formse(SE[s].δ[1])) for s in 1:nspec]...)
    write(io,[string("&",formse(SE[s].δ[2])) for s in 1:nspec]...) #automate this?
    write(io,"\\\\\n")
    write(io,repeat("&",4*nspec),"\\\\\n")

    # - Write factor share parameters
    write(io," & \\multicolumn{$nspec}{c}{\$\\tilde{\\phi}_{m}\$: Mother's Time} & \\multicolumn{$nspec}{c}{\$\\tilde{\\phi}_{f}\$: Father's Time} & \\multicolumn{$nspec}{c}{\$\\tilde{\\phi}_{x}\$: Childcare} & \\multicolumn{$nspec}{c}{\$\\phi_{\\theta}\$: TFP} ","\\\\\n")
    write(io,repeat(["&($s)" for s in 1:nspec],4)...,"\\\\",[midrule(s) for s in 1:4]...,"\n")

    vlist = union([s[specvar] for s in specs, specvar in [:vm,:vf,:vm,:vθ]]...)
    for v in vlist
        if v in keys(labels)
            vname = labels[v]
            vname = string(vname) #<-?
        else
            vname = string(v)
        end
        write(io,vname)
        # write estimates
        varlist = [:βm,:βf,:βy,:βθ]
        svarlist = [:vm,:vf,:vy,:vθ]#<- 
        for k in 1:4
            var = varlist[k]
            specvar = svarlist[k]
            for j in 1:nspec
                i = findfirst(specs[j][specvar].==v)
                if isnothing(i)
                    write(io,"&","-")
                else
                    xval = getfield(M[j],var)[i]
                    if var==:βθ
                        write(io,"&",form(xval))
                    else                        
                        pval = getfield(Pp[j],var)[i]
                        write(io,"&",format_pval(form,xval,pval))
                    end
                end
            end
        end
        write(io,"\\\\\n")
        # now write standard errors:
        
        for k in 1:4
            var = varlist[k]
            specvar = svarlist[k]
            for j in 1:nspec
                i = findfirst(specs[j][specvar].==v)
                if isnothing(i)
                    write(io,"&","-")
                else
                    write(io,"&",formse(getfield(SE[j],var)[i]))
                end
            end
        end
        write(io,"\\\\\n")
    end 
    write(io,"\\\\\n")
    write(io,"\\bottomrule")
    write(io,"\\end{tabular}")
    close(io)
end
function write_production_table_older(results,specs,labels,outfile::String)
    M = [update_older(results[r].est,specs[r],results[r].case) for r in eachindex(results)]
    SE = [update_older(results[r].se,specs[r],results[r].case) for r in eachindex(results)]
    Pp = [update_demand_older(results[r].p_indiv,specs[r]) for r in eachindex(results)]

    form(x) = @sprintf("%0.2f",x)
    formse(x) = string("(",@sprintf("%0.2f",x),")")
    nspec = length(M)
    
    midrule(s) = "\\cmidrule(r){$(2+(s-1)*nspec)-$(1+s*nspec)}"
    # Write the header:
    io = open(outfile, "w");
    write(io,"\\begin{tabular}{l",repeat("c",nspec*3),"}\\\\\\toprule","\n")

    # - work on elasticity parameters
    e_ρ = [1/(1-M[s].ρ) for s in 1:nspec]
    se_ρ = [SE[s].ρ/(1-M[s].ρ)^2 for s in 1:nspec]
    write(io," & \\multicolumn{$nspec}{c}{\$\\rho\$} & \\multicolumn{$nspec}{c}{\$\\delta_{1}\$} & \\multicolumn{$nspec}{c}{\$\\delta_{2}\$} ","\\\\\n")
    write(io,repeat(["&($s)" for s in 1:nspec],3)...,"\\\\",[midrule(s) for s in 1:3]...,"\n")
    
    # -- now write the estimates:
    
    write(io,[string("&",format_pval(form,e_ρ[s],Pp[s].ρ)) for s in 1:nspec]...)
    write(io,[string("&",form(M[s].δ[1])) for s in 1:nspec]...)
    write(io,[string("&",form(M[s].δ[2])) for s in 1:nspec]...) #automate this?
    write(io,"\\\\\n")
    write(io,[string("&",formse(se_ρ[s])) for s in 1:nspec]...)
    write(io,[string("&",formse(SE[s].δ[1])) for s in 1:nspec]...)
    write(io,[string("&",formse(SE[s].δ[2])) for s in 1:nspec]...) #automate this?
    write(io,"\\\\\n")
    write(io,repeat("&",3*nspec),"\\\\\n")

    # - Write factor share parameters
    write(io," & \\multicolumn{$nspec}{c}{\$\\tilde{\\phi}_{m}\$: Mother's Time} & \\multicolumn{$nspec}{c}{\$\\tilde{\\phi}_{f}\$: Father's Time} & \\multicolumn{$nspec}{c}{\$\\tilde{\\phi}_{\\theta}\$: TFP} ","\\\\\n")
    write(io,repeat(["&($s)" for s in 1:nspec],3)...,"\\\\",[midrule(s) for s in 1:3]...,"\n")

    vlist = union([s[specvar] for s in specs, specvar in [:vm,:vf,:vm,:vθ]]...)
    for v in vlist
        if v in keys(labels)
            vname = labels[v]
            vname = string(vname) #<-?
        else
            vname = string(v)
        end
        write(io,vname)
        # write estimates
        varlist = [:βm,:βf,:βθ]
        svarlist = [:vm,:vf,:vθ]#<- I'm an idiot for calling these different things
        for k in 1:3
            var = varlist[k]
            specvar = svarlist[k]
            for j in 1:nspec
                i = findfirst(specs[j][specvar].==v)
                if isnothing(i)
                    write(io,"&","-")
                else
                    xval = getfield(M[j],var)[i]
                    if var==:βθ
                        write(io,"&",form(xval))
                    else                        
                        pval = getfield(Pp[j],var)[i]
                        write(io,"&",format_pval(form,xval,pval))
                    end
                end
            end
        end
        write(io,"\\\\\n")
        # now write standard errors:
        
        for k in 1:3
            var = varlist[k]
            specvar = svarlist[k]
            for j in 1:nspec
                i = findfirst(specs[j][specvar].==v)
                if isnothing(i)
                    write(io,"&","-")
                else
                    write(io,"&",formse(getfield(SE[j],var)[i]))
                end
            end
        end
        write(io,"\\\\\n")
    end 
    write(io,"\\\\\n")
    write(io,"\\bottomrule")
    write(io,"\\end{tabular}")
    close(io)

end


# this function writes a table that summarizes the results of the unrestricted estimation from one specifications.
# Don't bother with the elasticities for now?
function write_production_table_unrestricted(res,spec,labels,outfile::String)
    P1,P2 = update_relaxed(res.est,spec,res.unrestricted,res.case)
    SE1,SE2 = update_relaxed(res.se,spec,res.unrestricted,res.case)
    Pu = update_demand(res.unrestricted,spec)

    form(x) = @sprintf("%0.2f",x)
    formse(x) = string("(",@sprintf("%0.2f",x),")")
    
    midrule(s) = "\\cmidrule(r){$(2+(s-1)*2)-$(1+s*2)}"
    # Write the header:
    io = open(outfile, "w");
    write(io,"\\begin{tabular}{lccccccc}","\\toprule","\n")
    #write(io,"\\begin{tabular}{l",repeat("c",nspec*4),"}\\\\\\toprule","\n")

    # - work on elasticity parameters
    write(io," & \\multicolumn{2}{c}{\$\\epsilon_{\\tau,g}\$} & \\multicolumn{2}{c}{\$\\epsilon_{x,H}\$} & {\$\\delta_{1}\$} & {\$\\delta_{2}\$} & \$2N(Q_{N} - \\tilde{Q}_{N})\$ ","\\\\\n")
    write(io," & Rel. Dem. & Prod. & Rel. Dem. & Prod. & - & - & - \\\\","\\cmidrule(r){2-3}","\\cmidrule(r){4-5}","\\cmidrule(r){6-6}","\\cmidrule(r){7-7}","\\cmidrule(r){8-8}","\n")
    
    # -- now write the estimates:
    
    write(io,"&",form(1/(1-P1.ρ)))
    if Pu.ρ
        write(io,"&",form(1/(1-P2.ρ)))
    else
        write(io,"& - ")
    end
    write(io,"&",form(1/(1-P1.γ)))
    if Pu.γ
        write(io,"&",form(1/(1-P2.γ)))
    else
        write(io,"& - ")
    end
    write(io,"&",form(P2.δ[1]),"&",form(P2.δ[2]),"&",form(res.DM),"\\\\\n")

    # ----- standard errors:
    write(io,"&",formse(SE1.ρ/(1-P1.ρ)^2))
    if Pu.ρ
        write(io,"&",formse(SE2.ρ/(1-P2.ρ^2)))
    else
        write(io,"& - ")
    end
    write(io,"&",formse(SE1.γ/(1-P1.γ)^2))
    if Pu.γ
        write(io,"&",formse(SE2.γ/(1-P2.γ)^2))
    else
        write(io,"& - ")
    end
    write(io,"&",formse(SE2.δ[1]),"&",formse(SE2.δ[2]), "&", formse(res.p_val),"\\\\\n")
    
    write(io,"\\\\\n")
    write(io,repeat("&",7),"\\\\\n")

    # - Write factor share parameters
    write(io," & \\multicolumn{2}{c}{\$\\tilde{\\phi}_{m}\$: Mother's Time} & \\multicolumn{2}{c}{\$\\tilde{\\phi}_{f}\$: Father's Time} & \\multicolumn{2}{c}{\$\\tilde{\\phi}_{x}\$: Childcare} &{\$\\phi_{\\theta}\$: TFP} ","\\\\\n")
    write(io," & Rel. Dem. & Prod. & Rel. Dem. & Prod. & Rel. Dem. & Prod. & -  \\\\","\\cmidrule(r){2-3}","\\cmidrule(r){4-5}","\\cmidrule(r){6-7}","\\cmidrule(r){8-8}","\n")

    vlist = union([spec[specvar] for specvar in [:vm,:vf,:vm,:vθ]]...)
    for v in vlist
        if v in keys(labels)
            vname = labels[v]
            vname = string(vname) #<-?
        else
            vname = string(v)
        end
        write(io,vname)
        # write estimates
        varlist = [:βm,:βf,:βy,:βθ]
        svarlist = [:vm,:vf,:vy,:vθ]#<- 
        for k in 1:4
            var = varlist[k]
            specvar = svarlist[k]
            i = findfirst(spec[specvar].==v)
            if isnothing(i)
                if var==:βθ
                    write(io,"&","-")
                else
                    write(io,"& - & -")
                end
            else
                if var==:βθ
                    xval = getfield(P2,var)[i]
                    write(io,"&",form(xval))
                else
                    xval = getfield(P1,var)[i]
                    write(io,"&",form(xval))
                    if getfield(Pu,var)[i]
                        xval = getfield(P2,var)[i]
                        write(io,"&",form(xval))
                    else
                        write(io,"& -") 
                    end
                end
            end
        end
        write(io,"\\\\\n")
        # now write standard errors:
        for k in 1:4
            var = varlist[k]
            specvar = svarlist[k]
            i = findfirst(spec[specvar].==v)
            if isnothing(i)
                if var==:βθ
                    write(io,"&") #,"-")
                else
                    write(io," & &")
                end
            else
                if var==:βθ
                    xval = getfield(SE2,var)[i]
                    write(io,"&",formse(xval))
                else
                    xval = getfield(SE1,var)[i]
                    write(io,"&",formse(xval))
                    if getfield(Pu,var)[i]
                        xval = getfield(SE2,var)[i]
                        write(io,"&",formse(xval))
                    else
                        write(io,"&") # -") 
                    end
                end
            end
        end
        write(io,"\\\\\n")
    end 
    write(io,"\\\\\n")
    write(io,"\\bottomrule")
    write(io,"\\end{tabular}")
    close(io)

end

# ---- this function writes a table to present all of the results from demand estimation

function write_demand_table(results,specs,labels,outfile::String)
    M = [update_demand(results[r].est,specs[r]) for r in eachindex(results)]
    SE = [update_demand(results[r].se,specs[r]) for r in eachindex(results)]

    form(x) = @sprintf("%0.2f",x)
    formse(x) = string("(",@sprintf("%0.2f",x),")")
    nspec = length(M)
    
    midrule(s) = "\\cmidrule(r){$(2+(s-1)*nspec)-$(1+s*nspec)}"
    # Write the header:
    io = open(outfile, "w");
    write(io,"\\begin{tabular}{l",repeat("c",nspec*4),"}\\\\\\toprule","\n")

    # - work on elasticity parameters and residual correlation test
    write(io," & \\multicolumn{$nspec}{c}{\$\\epsilon_{\\tau,g}\$} & \\multicolumn{$nspec}{c}{\$\\epsilon_{x,H}\$} & \\multicolumn{$nspec}{c}{Correl. residuals}","\\\\\n")
    write(io,repeat(["&($s)" for s in 1:nspec],3)...,"\\\\",[midrule(s) for s in 1:3]...,"\n")
    
    # -- now write the estimates:
    
    e_ρ = [1/(1-M[s].ρ) for s in 1:nspec]
    se_ρ = [SE[s].ρ/(1-M[s].ρ)^2 for s in 1:nspec]
    e_γ = [1/(1-M[s].γ) for s in 1:nspec]
    se_γ = [SE[s].γ/(1-M[s].γ)^2 for s in 1:nspec]

    write(io,[string("&",form(e_ρ[s])) for s in 1:nspec]...)
    write(io,[string("&",form(e_γ[s])) for s in 1:nspec]...)
    write(io,[string("&",form(results[s].pval)) for s in 1:nspec]...)
    write(io,["&" for s in 1:nspec]...) #automate this?
    write(io,"\\\\\n")
    write(io,[string("&",formse(se_ρ[s])) for s in 1:nspec]...)
    write(io,[string("&",formse(se_γ[s])) for s in 1:nspec]...)
    write(io,["&" for s in 1:nspec]...)
    write(io,["&" for s in 1:nspec]...) #automate this?
    write(io,"\\\\\n")
    write(io,repeat("&",3*nspec),"\\\\\n")

    # - Write factor share parameters
    write(io," & \\multicolumn{$nspec}{c}{\$\\tilde{\\phi}_{m}\$: Mother's Time} & \\multicolumn{$nspec}{c}{\$\\tilde{\\phi}_{f}\$: Father's Time} & \\multicolumn{$nspec}{c}{\$\\tilde{\\phi}_{x}\$: Childcare}","\\\\\n")
    write(io,repeat(["&($s)" for s in 1:nspec],3)...,"\\\\",[midrule(s) for s in 1:3]...,"\n")

    vlist = union([s[specvar] for s in specs, specvar in [:vm,:vf,:vm]]...)
    for v in vlist
        if v in keys(labels)
            vname = labels[v]
            vname = string(vname) #<-?
        else
            vname = string(v)
        end
        write(io,vname)
        # write estimates
        varlist = [:βm,:βf,:βy]
        svarlist = [:vm,:vf,:vy]#<- 
        for k in 1:3
            var = varlist[k]
            specvar = svarlist[k]
            for j in 1:nspec
                i = findfirst(specs[j][specvar].==v)
                if isnothing(i)
                    write(io,"&","-")
                else
                    write(io,"&",form(getfield(M[j],var)[i]))
                end
            end
        end
        write(io,"\\\\\n")
        # now write standard errors:
        
        for k in 1:3
            var = varlist[k]
            specvar = svarlist[k]
            for j in 1:nspec
                i = findfirst(specs[j][specvar].==v)
                if isnothing(i)
                    write(io,"&","-")
                else
                    write(io,"&",formse(getfield(SE[j],var)[i]))
                end
            end
        end
        write(io,"\\\\\n")
    end 
    write(io,"\\\\\n")
    write(io,"\\bottomrule")
    write(io,"\\end{tabular}")
    close(io)
end