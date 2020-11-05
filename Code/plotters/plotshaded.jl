include("headers.jl")

# rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
# fs = 10;
# plt.rcParams["font.size"] = fs
PyPlot.close("all")

pgfsettings("sans-serif")
increasefontsize()

d = LinRange(0,0.5,100)
r = LinRange(0, 1, 100)
fig1 = PyPlot.figure("Rate-distortion bound")
R1 = rdb.(d)
R2 = 1 .- 2*d
PyPlot.plot(R1, d, label="Lower bound")
PyPlot.plot(R2, d, label="Naive compression")
plt.:legend()


PyPlot.text(.7,.3,"Trivial")
PyPlot.text(.28,.23, "Feasible")
PyPlot.text(.08,.08,"Unfeasible")
PyPlot.fill_between(R2, 0.5, (1 .- R2)/2, color="grey", alpha=0.25, label="Trivial")
PyPlot.fill_betweenx(d, R1, R2, color="grey", alpha=0.05 , label="Feasible")
PyPlot.fill_betweenx(d, 0, R1, color="grey", alpha=0.5 , label="Unfeasible")

ax = gca()
ax.set_xlabel("Rate")
ax.set_ylabel("Distortion")

plt.tight_layout()

plt.savefig("../../Latex/images/rdb2.pgf", bbox_inches="tight")
