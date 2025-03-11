using DataFrames, MixedModels, StatsModels, Statistics

dyestuff = MixedModels.dataset(:dyestuff)

describe(DataFrame(dyestuff))


fm = @formula(yield ~ 1 + (1|batch))

fm1 = fit(MixedModel, fm, dyestuff)

fm1reml = fit(MixedModel, fm, dyestuff, REML=true)


sleepstudy = MixedModels.dataset(:sleepstudy)
fm2 = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days|subj)), sleepstudy)


for i in ranef(fm1reml)[1]
    print(i + 1527.5, "\n") 
end

g = DataFrame(dyestuff)

a = groupby(g, "batch")



[mean(i.yield) for i in a] 