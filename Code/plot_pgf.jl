# using PyPlot
function pgfsettings(font_family::String="serif")
    plt.rcdefaults()
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    rcParams["pgf.rcfonts"] = false
    rcParams["backend"] = "pgf"
    rcParams["text.usetex"] = true
    rcParams["pgf.texsystem"] = "xelatex"
    rcParams["font.family"] = font_family
    # rcParams["font.serif"] = ""
    return nothing
end

function increasefontsize(k::Real=2)
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    rcParams["font.size"] = 10.0*k
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
