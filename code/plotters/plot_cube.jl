include("headers.jl")

x = [1 1 1 1 0 0 0 0;
     1 1 0 0 1 1 0 0;
     1 0 1 0 1 0 1 0]


pgfsettings()
increasefontsize(2)
plt.close("all")
scatter3D(x[1,:], x[2,:], x[3,:], s=150, depthshade=false)
ax = gca()
lims = (0,1.2)
ax.set_xticks([0,1]); ax.set_xlabel("x1"); ax.set_xlim((0,1.2))
ax.set_yticks([0,1]); ax.set_ylabel("x2"); ax.set_ylim((-0.2,1))
ax.set_zticks([0,1]); ax.set_zlabel("x3"); ax.set_zlim((0,1.2))
plt.tight_layout()


n = 2
lw = 3
lw2 = 1

ax.plot(0:1, ones(n), ones(n), "b--", lw=lw2)
ax.plot(0:1, zeros(n), ones(n), "b--", lw=lw2)
ax.plot(0:1, ones(n), zeros(n), "b--", lw=lw2)
ax.plot(0:1, zeros(n), zeros(n), "b--", lw=lw2)

ax.plot( ones(n),0:1, ones(n), "b--", lw=lw2)
ax.plot( zeros(n),0:1, ones(n), "b--", lw=lw2)
ax.plot( ones(n), 0:1,zeros(n), "b--", lw=lw2)
ax.plot( zeros(n), 0:1,zeros(n), "b--", lw=lw2)

ax.plot( ones(n), ones(n),0:1, "b--", lw=lw2)
ax.plot( zeros(n), ones(n),0:1, "b--", lw=lw2)
ax.plot( ones(n), zeros(n),0:1, "b--", lw=lw2)
ax.plot( zeros(n), zeros(n),0:1, "b--", lw=lw2)

plt.savefig("./images/hypercube.pgf")

ax.plot(ones(n), ones(n), LinRange(0,1,n), "r", lw=lw)
ax.plot(ones(n), LinRange(0,1,n), ones(n), "r", lw=lw)
ax.plot(LinRange(0,1,n), zeros(n),  ones(n), "r", lw=lw)
plt.savefig("./images/hypercube2.pgf")
