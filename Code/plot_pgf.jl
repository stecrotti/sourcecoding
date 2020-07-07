# using PyPlot
function pgfsettings()
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    # rcParams["pgf.rcfonts"] = false
    rcParams["backend"] = "pgf"
    rcParams["text.usetex"] = true
    rcParams["pgf.texsystem"] = "xelatex"
    rcParams["font.family"] = "serif"
    # rcParams["font.serif"] = ""

    return nothing
end

function doublefontsize()
    pgfsettings()
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    # rcParams["pgf.preamble"] = raw"\setmainfont{TeX Gyre Pagella}[Scale=2]"
    rcParams["font.size"] = 20.0
    return nothing
end

function mpldefault()
    plt.rcdefaults()
    return nothing
end


# PyPlot.close("all")
# plt.ioff()
# PyPlot.plot([1,2,3], [4,5,6], label="bu")
# PyPlot.legend()
# plt.:xlabel("X label")
# plt.:ylabel("Y label")

# plt.savefig("../../../../provetex/test.pgf", format="pgf")
