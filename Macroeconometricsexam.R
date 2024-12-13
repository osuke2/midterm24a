require(tseries)
require(forecast)
# Cleaning previous data
rm(list=ls())
ls()

# Read xls data
Data_SAF_1_
AD_shock_Q_1_

# Check stationarity
par(mfrow=c(3,2))
ts.plot(Data_SAF_1_[,2],ylab="y")
ts.plot(Data_SAF_1_[,3],ylab="Dp")
ts.plot(Data_SAF_1_[,4],ylab="eq")
ts.plot(Data_SAF_1_[,5],ylab="ep")
ts.plot(Data_SAF_1_[,6],ylab="r")
ts.plot(Data_SAF_1_[,7],ylab="lr")

acf(Data_SAF_1_[,2])
acf(Data_SAF_1_[,3])
acf(Data_SAF_1_[,4])
acf(Data_SAF_1_[,5])
acf(Data_SAF_1_[,6])
acf(Data_SAF_1_[,7])

# Check breaks
par(mfrow=c(1,1))
SAF_GDP=Data_SAF_1_[,2]
SAF_GDP.ts=ts(SAF_GDP,freq=4,start=c(1979,2))
ts.plot(SAF_GDP.ts)

# ARmodel

# estimate AR(p)
SAF_data=Data_SAF_1_[56:151,]
SAF_GDP2.ts=ts(SAF_data[,2],freq=4,start=c(1993,1))
SAF_Growthrate <- diff(SAF_GDP2.ts,lag=4)*100
ts.plot(SAF_Growthrate)
acf(SAF_Growthrate)
pacf(SAF_Growthrate)
jarque.bera.test(SAF_Growthrate)
ar.SAF=ar(SAF_Growthrate,AIC=T)
plot(ar.SAF$aic,type="l",ylab="AIC")

# We choose a AR(3)
ar3 <- arima(SAF_Growthrate, order = c(3, 0, 0), include.mean = T)
summary(ar3)
print(ar3)
horiz2=12
predict(ar3,n.ahead=horiz2)

# MSE
SAF_Growthrate_hat <- SAF_Growthrate - residuals(ar3)
residuals <- residuals(ar3)
MSE <- mean(residuals^2)
print(MSE)
Box.test(residuals,lag=1)
par(mfrow=c(2,1))
ts.plot(SAF_Growthrate_hat,ylab="forecast")
ts.plot(SAF_Growthrate,ylab="actual")

# VARmodel
# estimate VAR(p)

require(vars)
# Package to run HP filters
require("mFilter")

par(mfrow=c(1,1))


SAF_aux <- hpfilter(SAF_data[,3],freq=129600)
SAF_Dp <- SAF_aux$cycle[5:96]
ts.plot(SAF_Dp)
pacf(SAF_Dp)

SAF_aux <- hpfilter(SAF_data[,4],freq=129600)
SAF_eq <- SAF_aux$cycle[5:96]
ts.plot(SAF_eq)
pacf(SAF_eq)

SAF_aux <- hpfilter(SAF_data[,5],freq=129600)
SAF_ep <- SAF_aux$cycle[5:96]
ts.plot(SAF_ep)
pacf(SAF_ep)

SAF_aux <- hpfilter(SAF_data[,6],freq=129600)
SAF_r <- SAF_aux$cycle[5:96]
ts.plot(SAF_r)
pacf(SAF_r)

SAF_aux <- hpfilter(SAF_data[,7],freq=129600)
SAF_lr <- SAF_aux$cycle[5:96]
ts.plot(SAF_lr)
pacf(SAF_lr)


set1=cbind(SAF_Growthrate,SAF_Dp,SAF_eq,SAF_ep,SAF_r,SAF_lr)

mod.est1=VAR(set1,p=1)
logLik(mod.est1)

# Granger Causality
causality(mod.est1,cause="SAF_Dp")$Granger
causality(mod.est1,cause="SAF_eq")$Granger
causality(mod.est1,cause="SAF_ep")$Granger
causality(mod.est1,cause="SAF_r")$Granger
causality(mod.est1,cause="SAF_lr")$Granger

set2=cbind(SAF_Growthrate,SAF_ep,SAF_lr)

# Loglikelihood Ratio test
mod.est1=VAR(set2,p=1)
logLik(mod.est1)
mod.est2=VAR(set2,p=2)
logLik(mod.est2)
mod.est3=VAR(set2,p=3)
logLik(mod.est3)

d1f12=2*(logLik(mod.est2)-logLik(mod.est1))
d1f12
qchisq(p=0.99,df=9)
1-pchisq(q=d1f12,df=9)

d1f23=2*(logLik(mod.est3)-logLik(mod.est2))
d1f23
qchisq(p=0.99,df=9)
1-pchisq(q=d1f23,df=9)

# We choose a VAR(2)
summary(mod.est2)
forecast_result <- predict(mod.est2,n.ahead=horiz2)
print(forecast_result)

# MSE
SAF_hat <- fitted(mod.est2)
SAF_Growthrate_hathat <- SAF_hat[,1]
SAF_Growthrate_actual <- SAF_Growthrate[3:92]
residuals2 <- SAF_Growthrate_actual - SAF_Growthrate_hathat
MSE2 <- mean(residuals2^2)
print(MSE2)
Box.test(residuals2,lag=1)
par(mfrow=c(2,1))
ts.plot(SAF_Growthrate_hathat,ylab="forecast")
ts.plot(SAF_Growthrate_actual,ylab="actual")

#SVAR model

# Reorder variables according to structure for SVAR estimation
Data_SAF_1_p <- Data_SAF_1_[c(6,7,5,3,2,4)]

# Central bank rate
SAF_aux <- hpfilter(Data_SAF_1_p[,1],freq=129600)
Data_SAF_1_p[,1] <- SAF_aux$cycle

# 10 years rate
SAF_aux <- hpfilter(Data_SAF_1_p[,2],freq=129600)
Data_SAF_1_p[,2] <- SAF_aux$cycle

# exchange rate
SAF_aux <- hpfilter(Data_SAF_1_p[,3],freq=129600)
Data_SAF_1_p[,3] <- SAF_aux$cycle

# Delta-log of CPI
SAF_aux <- hpfilter(Data_SAF_1_p[,4],freq=129600)
Data_SAF_1_p[,4] <- SAF_aux$cycle

# GDP in log-level
SAF_aux <- hpfilter(Data_SAF_1_p[,5],freq=129600)
Data_SAF_1_p[,5] <- SAF_aux$cycle

# equity price in log-level
SAF_aux <- hpfilter(Data_SAF_1_p[,6],freq=129600)
Data_SAF_1_p[,6] <- SAF_aux$cycle
summary(Data_SAF_1_p)

# VAR Estimation
set3=cbind(Data_SAF_1_p[,1],Data_SAF_1_p[,2],Data_SAF_1_p[,3],Data_SAF_1_p[,4],Data_SAF_1_p[,5],Data_SAF_1_p[,6])

SAF.est1=VAR(set3,p=1)
logLik(SAF.est1)

SAF.est2=VAR(set3,p=2)
logLik(SAF.est2)

SAF_dif12=2*(logLik(SAF.est2)-logLik(SAF.est1))
qchisq(p=0.99,df=36)
1-pchisq(q=SAF_dif12,df=36)

SAF.est3=VAR(set3,p=3)
logLik(SAF.est3)

SAF_dif23=2*(logLik(SAF.est3)-logLik(SAF.est2))
qchisq(p=0.99,df=36)
SAF_dif23
1-pchisq(q=SAF_dif23,df=36)

summary(SAF.est2)
VARselect(set3,lag.max=8)

# We choose a VAR(2)
Data_SAF_results <- VAR(Data_SAF_1_p,p=2)

#plot IRF
Data_SAF_results.irf <- irf(Data_SAF_results, response = "lr", impulse = "r", n.ahead = 36, boot = TRUE)
plot(Data_SAF_results.irf)

Data_SAF_results.irf <- irf(Data_SAF_results, response = "ep", impulse = "r", n.ahead = 36, boot = TRUE)
plot(Data_SAF_results.irf)

Data_SAF_results.irf <- irf(Data_SAF_results, response = "Dp", impulse = "r", n.ahead = 36, boot = TRUE)
plot(Data_SAF_results.irf)

Data_SAF_results.irf <- irf(Data_SAF_results, response = "y", impulse = "r", n.ahead = 36, boot = TRUE)
plot(Data_SAF_results.irf)

Data_SAF_results.irf <- irf(Data_SAF_results, response = "eq", impulse = "r", n.ahead = 36, boot = TRUE)
plot(Data_SAF_results.irf)

# Local Projection
require(lpirfs)

#exogenous data
par(mfrow=c(1,1))
AD_shock_Q_1.ts=ts(AD_shock_Q_1_[9:96,2],freq=4,start=c(1985,1))
ts.plot(AD_shock_Q_1.ts)
acf(AD_shock_Q_1.ts)


AD_shock_Q_1_lim <- AD_shock_Q_1_[9:96,2]

#endogenous data
SAF_data_lim <- SAF_data[24:111,2:7]

# Estimate linear model 
results_lin_iv <- lp_lin_iv(endog_data = SAF_data_lim, lags_endog_lin = 4, shock = AD_shock_Q_1_lim, trend = 0, confint = 1.96, hor = 20)
# Make and save linear plots 
plot(results_lin_iv)