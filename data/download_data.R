library('survival')

data("diabetic")
a = data.frame(diabetic)

write.csv(diabetic, 'diabetic')

# b = a[-c(21),] 
# mfit <- coxph(Surv(time, status) ~ sex + pspline(age, df=3), data=kidney)
# mfit
# termplot(mfit, term=2, se=TRUE, col.term=1, col.se=1)
# 
# ptemp <- termplot(mfit, se=TRUE, plot=FALSE)
# attributes(ptemp)



mfit <- coxph(Surv(time, status) ~ . , data=diabetic)
mfit
# termplot(mfit, term=2, se=TRUE, col.term=1, col.se=1)
# 
# ptemp <- termplot(mfit, se=TRUE, plot=FALSE)
# attributes(ptemp)





# mfit <- coxph(Surv(futime, death) ~ sex + pspline(age, df=20), data=mgus)
# mfit
# 
# termplot(mfit, term=2, se=TRUE, col.term=1, col.se=1)
# 
# ptemp <- termplot(mfit, se=TRUE, plot=FALSE)
# attributes(ptemp)