include("FactorGraph.jl")
include("linear_algebra_gfq.jl")
include("ldpc_graph.jl")
include("convolutions.jl")
include("LossyModel.jl")
include("gfrbp.jl")
include("simanneal.jl")
include("exhaustive_enum.jl")
include("optimal_cycle.jl")
include("Simulation.jl")
include("./plotters/approx_entropy.jl")
include("./plotters/plot_pgf.jl")


# Send notifications to telegram when program execution ends
cwd = pwd()
cd(@__DIR__)
notifications_path = "../../telegram/notifications.jl"
if isfile(notifications_path)
    include(notifications_path)
    function send_notification(args...; kwargs...)
        send_notif(args...; kwargs...)
    end
else
    function send_notification(args...; kwargs...)
        nothing
    end
end
cd(cwd)