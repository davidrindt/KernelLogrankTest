library('survival')

data("leukemia")
a = data.frame(leukemia)

write.csv(leukemia, 'leukemia')

# b = a[-c(21),] 
# mfit <- coxph(Surv(time, status) ~ sex + pspline(age, df=3), data=kidney)
# mfit
# termplot(mfit, term=2, se=TRUE, col.term=1, col.se=1)
# 
# ptemp <- termplot(mfit, se=TRUE, plot=FALSE)
# attributes(ptemp)



mfit <- coxph(Surv(time, status) ~ trt + pspline(risk, df=0), data=diabetic)
mfit
termplot(mfit, term=2, se=TRUE, col.term=1, col.se=1)

ptemp <- termplot(mfit, se=TRUE, plot=FALSE)
attributes(ptemp)





# mfit <- coxph(Surv(futime, death) ~ sex + pspline(age, df=20), data=mgus)
# mfit
# 
# termplot(mfit, term=2, se=TRUE, col.term=1, col.se=1)
# 
# ptemp <- termplot(mfit, se=TRUE, plot=FALSE)
# attributes(ptemp)