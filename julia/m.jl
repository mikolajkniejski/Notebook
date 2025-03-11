using CSV
using DataFrames
using Plots
using Statistics
using Distributions
using StatsPlots

a = CSV.File("housing.csv") |> DataFrame

function normalize(sorted)
    sorted = sorted .- mean(sorted)
    sorted = sorted ./ std(sorted)
    return sorted
end


a.logSalePrice = log.(a.SalePrice)


sorted = sort(a.logSalePrice)  
sorted = normalize(sorted)

prbs = [(i - 0.5) / length(a.Order) for i in a.Order]
th = quantile(Normal(0.0, 1.0), prbs) 



probs = range(0.000, 1, length=999)  # 999 points from 0.001 to 0.999
x = quantile(Normal(0.0, 1.0), probs)    # Theoretical quantiles (X-axis)
y = quantile(sorted, probs)

plot(scatter(th, prbs))



plot(scatter(sorted, th), qqplot(y, x))
