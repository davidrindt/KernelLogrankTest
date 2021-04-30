library(survival)
library(condSURV)
library(JM)
library(dplyr)
library(survminer)
library(clustcurv)
library(ggplot2)

# First we start with loading the dataset
data("colon", package = "survival")
write.csv(colon, 'colon')

# Then we munge it according to ?boot::melanoma
library(dplyr)
library(magrittr)
# melanoma %<>% 
#   mutate(status = factor(status,
#                          levels = 1:3,
#                          labels = c("Died from melanoma", 
#                                     "Alive", 
#                                     "Died from other causes")),
#          ulcer = factor(ulcer,
#                         levels = 0:1,
#                         labels = c("Absent", "Present")),
#          time = time/365.25, # All variables should be in the same time unit
#          sex = factor(sex,
#                       levels = 0:1,
#                       labels = c("Female", "Male")))
# write.csv(melanoma, 'melanoma')

# 
# data("leukemia")
# a = data.frame(leukemia)
# write.csv(leukemia, 'leukemia')
# web <- "https://s3.amazonaws.com/udacity-hosted-downloads/ud651/prosperLoanData.csv"
# loan <- read.csv(web)
# 
# # loan_nd <- loan[unique(loan$LoanKey), ] 
# loan_nd <- distinct(loan, LoanKey)
# loan_nd <- loan[!duplicated(loan$LoanKey), ]
# # loan_nd <- loan[unique(loan$LoanKey), ] 
# 
# # removing LoanStatus no needed 
# sel_status  <- loan_nd$LoanStatus %in% c("Completed", "Current", 
#                                          "ChargedOff", "Defaulted", 
#                                          "Cancelled")
# loan_filtered <- loan_nd[sel_status, ]
# 
# # creating status variable for censoring
# loan_filtered$status <- ifelse(
#   loan_filtered$LoanStatus == "Defaulted" |
#     loan_filtered$LoanStatus == "Chargedoff",  1, 0)
# 
# # adding the final date to "current" status
# head(levels(loan_filtered$ClosedDate))
# ## [1] ""                    "2005-11-25 00:00:00" "2005-11-29 00:00:00"
# ## [4] "2005-11-30 00:00:00" "2005-12-08 00:00:00" "2005-12-28 00:00:00"
# levels(loan_filtered$ClosedDate)[1] <- "2014-11-03 00:00:00"
# 
# # creating the time-to-event variable
# loan_filtered$start <- as.Date(loan_filtered$LoanOriginationDate)
# loan_filtered$end <- as.Date(loan_filtered$ClosedDate)
# loan_filtered$time <- as.numeric(difftime(loan_filtered$end, loan_filtered$start, units = "days"))
# 
# # there is an error in the data (time to event less than 0)
# loan_filtered <- loan_filtered[-loan_filtered$time < 0, ]
# 
# # just considering a year of loans creation
# ii <- format(as.Date(loan_filtered$LoanOriginationDate),'%Y') %in% c("2006")
# loan_filtered <- loan_filtered[ii, ] 
# 
# loan_filtered$LoanOriginalAmount2 <-  loan_filtered$LoanOriginalAmount/10000
# 
# 
# to_save_data <- loan_filtered[, c("IsBorrowerHomeowner", "LoanOriginalAmount2", 'time', 'status')]
# 
# write.csv(to_save_data, 'loan_data')

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