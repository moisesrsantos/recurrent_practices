library(forecast)

data = read.table("data/stockmarket_f.data")

data = ts(data)

fit = auto.arima(data)
ret = fit$fitted

write.table(ret,"data/model.data", row.names = F, col.names = F, dec=".", sep=",")


plot(data)
lines(ret,col = "red")