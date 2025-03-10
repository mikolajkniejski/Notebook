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
a.prob = a.Order .- 0.5 ./ length(a.SalePrice)
a.prob2 = quantile(Normal(0.0, 1.0), a.prob)
sorted = sort(a.SalePrice)  
sorted = normalize(sorted)



y = quantile(sorted, [i / 1000 for i in range(1, 999)])
x = quantile(Normal(0.0, 1.0), [i / 1000 for i in range(1, 999)])

plot(scatter(x, y), qqplot( rand(Normal(), 1000), sorted))

