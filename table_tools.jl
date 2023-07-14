# ----------- Tools for writing results to file

# this function writes to file the value of parameter v in for each element of the vector of results M. This may or may not be a vector.
function write_line!(io,format,M,v::Symbol,i::Int=0,vname::String="",endline=true)
    nspec = length(M)
    write(io,vname,"&")
    for s in 1:nspec
        if i==0
            write(io,format(getfield(M[s],v)),"&")
        else
            write(io,format(getfield(M[s],v)[i]),"&")
        end
    end
    if endline
        write(io,"\\\\","\n")
    end
end

# this function does the same as above but includes p-values
function write_pars!(io,format,M,Pv,v::Symbol,i::Int=0,vname::String="",endline = true)
    nspec = length(M)
    write(io,vname,"&")
    for s in 1:nspec
        if i==0
            pval = getfield(Pv[s],v)
            pstr = pval_ind(pval)
            write(io,format(getfield(M[s],v)),pstr,"&")
        else
            pval = getfield(M[s],v)[i]
            pstr = pval_ind(pval)
            write(io,format(getfield(M[s],v)[i]),pstr,"&")
        end
    end
    if endline
        write(io,"\\\\","\n")
    end
end

function pval_ind(p,char = "*")
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

function format_pval(form,x,p)
    return string("\$",form(x),pval_ind(p),"\$")
end


function write_observables!(io,format,formatse,M,SE,specs,labels,var::Symbol,specvar::Symbol)
    nspec = length(M)
    vlist = union([s[specvar] for s in specs]...)
    for v in vlist
        if v in keys(labels)
            vname = labels[v]
            vname = string(vname) #<-?
        else
            vname = string(v)
        end
        write(io,vname,"&")
        # write estimates
        for j in 1:nspec
            i = findfirst(specs[j][specvar].==v)
            if isnothing(i)
                write(io,"-","&")
            else
                write(io,format(getfield(M[j],var)[i]),"&")
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
                write(io,formatse(getfield(SE[j],var)[i]),"&")
            end
        end
        write(io,"\\\\")

    end
end

function write_production_table(M,SE,Pp,specs,labels,outfile::String)
    form(x) = @sprintf("%0.2f",x)
    formse(x) = string("(",@sprintf("%0.2f",x),")")
    nspec = length(M)
    
    midrule(s) = "\\cmidrule(r){$(2+(s-1)*nspec)-$(1+s*nspec)}"
    # Write the header:
    io = open(outfile, "w");
    write(io,"\\begin{tabular}{l",repeat("c",nspec*4),"}\\\\\\toprule","\n")

    # - work on elasticity parameters
    write(io," & \\multicolumn{$nspec}{c}{\$\\rho\$} & \\multicolumn{$nspec}{c}{\$\\gamma\$} & \\multicolumn{$nspec}{c}{\$\\delta_{1}\$} & \\multicolumn{$nspec}{c}{\$\\delta_{2}\$} ","\\\\\n")
    write(io,repeat(["&($s)" for s in 1:nspec],4)...,"\\\\",[midrule(s) for s in 1:4]...,"\n")
    
    # -- now write the estimates:
    
    write(io,[string("&",format_pval(form,M[s].ρ,Pp[s].ρ)) for s in 1:nspec]...)
    write(io,[string("&",format_pval(form,M[s].γ,Pp[s].γ)) for s in 1:nspec]...)
    write(io,[string("&",form(M[s].δ[1])) for s in 1:nspec]...)
    write(io,[string("&",form(M[s].δ[2])) for s in 1:nspec]...) #automate this?
    write(io,"\\\\\n")
    write(io,[string("&",formse(SE[s].ρ)) for s in 1:nspec]...)
    write(io,[string("&",formse(SE[s].γ)) for s in 1:nspec]...)
    write(io,[string("&",formse(SE[s].δ[1])) for s in 1:nspec]...)
    write(io,[string("&",formse(SE[s].δ[2])) for s in 1:nspec]...) #automate this?
    write(io,"\\\\\n")
    write(io,repeat("&",4*nspec),"\\\\\n")

    # - Write factor share parameters
    write(io," & \\multicolumn{$nspec}{c}{\$\\phi_{m}\$: Mother's Time} & \\multicolumn{$nspec}{c}{\$\\phi_{f}\$: Father's Time} & \\multicolumn{$nspec}{c}{\$\\phi_{Y}\$: Childcare} & \\multicolumn{$nspec}{c}{\$\\phi_{\\theta}\$: TFP} ","\\\\\n")
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
        svarlist = [:vm,:vf,:vy,:vθ]#<- I'm an idiot for calling these different things
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

function write_production_table_older(M,SE,Pp,specs,labels,outfile::String)
    form(x) = @sprintf("%0.2f",x)
    formse(x) = string("(",@sprintf("%0.2f",x),")")
    nspec = length(M)
    
    midrule(s) = "\\cmidrule(r){$(2+(s-1)*nspec)-$(1+s*nspec)}"
    # Write the header:
    io = open(outfile, "w");
    write(io,"\\begin{tabular}{l",repeat("c",nspec*3),"}\\\\\\toprule","\n")

    # - work on elasticity parameters
    write(io," & \\multicolumn{$nspec}{c}{\$\\rho\$} & \\multicolumn{$nspec}{c}{\$\\delta_{1}\$} & \\multicolumn{$nspec}{c}{\$\\delta_{2}\$} ","\\\\\n")
    write(io,repeat(["&($s)" for s in 1:nspec],3)...,"\\\\",[midrule(s) for s in 1:3]...,"\n")
    
    # -- now write the estimates:
    
    write(io,[string("&",format_pval(form,M[s].ρ,Pp[s].ρ)) for s in 1:nspec]...)
    write(io,[string("&",form(M[s].δ[1])) for s in 1:nspec]...)
    write(io,[string("&",form(M[s].δ[2])) for s in 1:nspec]...) #automate this?
    write(io,"\\\\\n")
    write(io,[string("&",formse(SE[s].ρ)) for s in 1:nspec]...)
    write(io,[string("&",formse(SE[s].δ[1])) for s in 1:nspec]...)
    write(io,[string("&",formse(SE[s].δ[2])) for s in 1:nspec]...) #automate this?
    write(io,"\\\\\n")
    write(io,repeat("&",3*nspec),"\\\\\n")

    # - Write factor share parameters
    write(io," & \\multicolumn{$nspec}{c}{\$\\phi_{m}\$: Mother's Time} & \\multicolumn{$nspec}{c}{\$\\phi_{f}\$: Father's Time} & \\multicolumn{$nspec}{c}{\$\\phi_{\\theta}\$: TFP} ","\\\\\n")
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




function writetable(M,SE,specs,labels,pvals,outfile::String,production = false)
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
    if production
        for j in 1:2
            write_line!(io,form,M,:δ,j,"\$\\delta_{$j}\$")
            write_line!(io,formse,SE,:δ,j)
        end
        write_line!(io,form,M,:λ,0,"\$\\lambda_{AP}\$")
        write_line!(io,formse,SE,:λ,0) 
    end
    # δ_1 and δ_2 #are there delta parameters here?
    #for j in 1:2
    #    write_line!(io,form,M,:δ,j,"\$\\delta_{$j}\$")
    #    write_line!(io,formse,SE,:δ,j)
    #end

    # Factor share Parameters:
    # a_{m}
    # write the header:
    write(io,"& \\multicolumn{$(nspec+1)}{c}{\$\\phi_{m}\$: Mother's Time}\\\\\\cmidrule(r){2-$(nspec+2)}")
    write_observables!(io,form,formse,M,SE,specs,labels,:βm,:vm)
    # a_{f}
    write(io,"& \\multicolumn{$(nspec+1)}{c}{\$\\phi_{f}\$: Father's Time}\\\\\\cmidrule(r){2-$(nspec+2)}")
    write_observables!(io,form,formse,M,SE,specs,labels,:βf,:vf)
    # a_{g}
    write(io,"& \\multicolumn{$(nspec+1)}{c}{\$\\phi_{Y}\$: Childcare}\\\\\\cmidrule(r){2-$(nspec+2)}")
    write_observables!(io,form,formse,M,SE,specs,labels,:βy,:vy)

    # test results
    write(io,"& \\multicolumn{$(nspec+1)}{c}{Residual Correlation Test}\\\\\\cmidrule(r){2-$(nspec+2)}")
    write(io,"p-value","&")
    for s in 1:nspec
        write(io,form(pvals[s]),"&")
    end
    write(io,"\\\\","\n")


    write(io,"\\bottomrule")
    write(io,"\\end{tabular}")
    close(io)

end

function write_production_table_unrestricted(P1,P2,Pu,SE1,SE2,spec,labels,test_stat,p_val,outfile::String)
    form(x) = @sprintf("%0.2f",x)
    formse(x) = string("(",@sprintf("%0.2f",x),")")
    
    midrule(s) = "\\cmidrule(r){$(2+(s-1)*2)-$(1+s*2)}"
    # Write the header:
    io = open(outfile, "w");
    write(io,"\\begin{tabular}{lccccccc}","\\toprule","\n")
    #write(io,"\\begin{tabular}{l",repeat("c",nspec*4),"}\\\\\\toprule","\n")

    # - work on elasticity parameters
    write(io," & \\multicolumn{2}{c}{\$\\rho\$} & \\multicolumn{2}{c}{\$\\gamma\$} & {\$\\delta_{1}\$} & {\$\\delta_{2}\$} & \$2N(Q_{N} - \\tilde{Q}_{N})\$ ","\\\\\n")
    write(io," & (R) & (U) & (R) & (U) & - & - & - \\\\","\\cmidrule(r){2-3}","\\cmidrule(r){4-5}","\\cmidrule(r){6-6}","\\cmidrule(r){7-7}","\\cmidrule(r){8-8}","\n")
    
    # -- now write the estimates:
    
    write(io,"&",form(P1.ρ))
    if Pu.ρ
        write(io,"&",form(P2.ρ))
    else
        write(io,"& - ")
    end
    write(io,"&",form(P1.γ))
    if Pu.γ
        write(io,"&",form(P2.γ))
    else
        write(io,"& - ")
    end
    write(io,"&",form(P2.δ[1]),"&",form(P2.δ[2]),"&",form(test_stat),"\\\\\n")

    # ----- standard errors:
    write(io,"&",formse(SE1.ρ))
    if Pu.ρ
        write(io,"&",formse(SE2.ρ))
    else
        write(io,"& - ")
    end
    write(io,"&",formse(SE1.γ))
    if Pu.γ
        write(io,"&",formse(SE2.γ))
    else
        write(io,"& - ")
    end
    write(io,"&",formse(SE2.δ[1]),"&",formse(SE2.δ[2]), "&", formse(p_val),"\\\\\n")
    
    write(io,"\\\\\n")
    write(io,repeat("&",7),"\\\\\n")

    # - Write factor share parameters
    write(io," & \\multicolumn{2}{c}{\$\\phi_{m}\$: Mother's Time} & \\multicolumn{2}{c}{\$\\phi_{f}\$: Father's Time} & \\multicolumn{2}{c}{\$\\phi_{Y}\$: Childcare} &{\$\\phi_{\\theta}\$: TFP} ","\\\\\n")
    write(io," & (R) & (U) & (R) & (U) & (R) & (U) & -  \\\\","\\cmidrule(r){2-3}","\\cmidrule(r){4-5}","\\cmidrule(r){6-7}","\\cmidrule(r){8-8}","\n")

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
        svarlist = [:vm,:vf,:vy,:vθ]#<- I'm an idiot for calling these different things
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
