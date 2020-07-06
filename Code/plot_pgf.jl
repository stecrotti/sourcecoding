# using PyPlot
function pgfsettings()
    plt.rcdefaults()
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    rcParams["pgf.rcfonts"] = false
    rcParams["backend"] = "pgf"
    rcParams["text.usetex"] = true
    rcParams["pgf.texsystem"] = "xelatex"
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ""

    return nothing
end

# PyPlot.close("all")
# plt.ioff()
# PyPlot.plot([1,2,3], [4,5,6], label="bu")
# PyPlot.legend()
# plt.:xlabel("X label")
# plt.:ylabel("Y label")

# plt.savefig("../../../../provetex/test.pgf", format="pgf")
