using Plots

function det_heatmap(REheatres, IMheatres, REbounds, IMbounds, polydegs, lattice, parameters; imag_part=false, real_part=false)
    RErange = LinRange(REbounds[1],REbounds[2],REheatres)
    IMrange = LinRange(IMbounds[1],IMbounds[2],IMheatres)
    if imag_part == true
        heatMDet = [imag(det(getpolyxM(polydegs,x+y*im,lattice,parameters))) for y in IMrange, x in RErange]
        title_HM = string("Im(det(M)) ",2.99792458e5/parameters.lambda)
    elseif real_part == true
        heatMDet = [real(det(getpolyxM(polydegs,x+y*im,lattice,parameters))) for y in IMrange, x in RErange]
        title_HM = string("Re(det(M)) ",2.99792458e5/parameters.lambda)
    else
        heatMDet = [log(abs(det(getpolyxM(polydegs,x+y*im,lattice,parameters)))) for y in IMrange, x in RErange]
        title_HM = string("abs(det(M)) ",2.99792458e5/parameters.lambda)
    end
    return heatmap(RErange,IMrange,heatMDet,title=title_HM)
end

function eigenvalues_Mk(polydegs,manual_k,lattice,parameters)
    return eigen(getpolyxM(polydegs,manual_k,lattice,parameters))
end

function eval_heatmap(REheatres, IMheatres, REbounds, IMbounds, polydegs, lattice, parameters)
    RErange = LinRange(REbounds[1],REbounds[2],REheatres)
    IMrange = LinRange(IMbounds[1],IMbounds[2],IMheatres)
    heatMk_evals = [log(minimum(abs.(eigenvalues_Mk(polydegs,x+y*im,lattice,parameters).values))) for y in IMrange, x in RErange]
    return heatmap(RErange,IMrange,heatMk_evals,title=string("smallest eigenvalue of M(k) ",2.99792458e5/parameters.lambda))
end
