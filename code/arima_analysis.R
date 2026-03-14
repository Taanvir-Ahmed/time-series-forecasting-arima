# ============================================================
# Time Series Forecasting with ARIMA
# Series analyzed: V93
# Dataset: Case_study.csv
# Workflow:
# 1) Preliminary analysis
# 2) ARIMA model estimation and selection
# 3) Residual diagnostics
# 4) Out-of-sample forecasting
# ============================================================

setwd("/Users/tanvirahmed/Desktop/time-series-forecasting-arima")
suppressPackageStartupMessages({
  library(tseries)
  library(forecast)
  library(lmtest)
})

# -----------------------------
# 0. Load data
# -----------------------------
data <- read.csv('/Users/tanvirahmed/Desktop/time-series-forecasting-arima/data/Case_study.csv')
y <- data$V93

# In-sample and out-of-sample split
y_train <- y[1:100]
y_test  <- y[101:200]

y_ts <- ts(y_train)

# Create figures folder if it does not exist
dir.create("figures", showWarnings = FALSE)

# -----------------------------
# 1. Preliminary analysis
# -----------------------------

# 1.1 Time series plot
png("figures/series_plot.png", width = 900, height = 500)
plot(y_ts, type = "l",
     main = "Time Series Plot of V93",
     ylab = "Y_t",
     xlab = "Time")
dev.off()

# 1.2 Differencing analysis
dy  <- diff(y_ts)
d2y <- diff(y_ts, differences = 2)

png("figures/differencing_acf.png", width = 900, height = 1200)
par(mfrow = c(3, 1))
acf(y_ts, main = "ACF of Original Series")
acf(dy,  main = "ACF of First Difference")
acf(d2y, main = "ACF of Second Difference")
dev.off()

# ADF tests
adf_original <- adf.test(y_ts)
adf_diff1    <- adf.test(dy)

print(adf_original)
print(adf_diff1)

# 1.3 ACF/PACF of differenced series
png("figures/acf_pacf_differenced.png", width = 1000, height = 500)
par(mfrow = c(1, 2))
acf(dy,  main = "ACF of Differenced Series")
pacf(dy, main = "PACF of Differenced Series")
dev.off()

# -----------------------------
# 2. ARIMA model estimation and selection
# -----------------------------

results <- data.frame(
  p = integer(),
  q = integer(),
  AIC = double(),
  BIC = double()
)

for (p in 0:4) {
  for (q in 0:4) {
    if (!(p == 0 & q == 0) && !(p == 0) && !(q == 0)) {
      fit <- arima(y_ts, order = c(p, 1, q))
      results <- rbind(
        results,
        data.frame(
          p = p,
          q = q,
          AIC = AIC(fit),
          BIC = BIC(fit)
        )
      )
    }
  }
}

results_AIC <- results[order(results$AIC), ]
results_BIC <- results[order(results$BIC), ]

top3_AIC <- head(results_AIC, 3)
top3_BIC <- head(results_BIC, 3)

print(results)
print(results_AIC)
print(results_BIC)
print(top3_AIC)
print(top3_BIC)

dir.create("results", showWarnings = FALSE)

write.csv(results,
          "results/model_selection_results.csv",
          row.names = FALSE)

# Best three models
model1 <- arima(y_ts, order = c(1, 1, 3))
model2 <- arima(y_ts, order = c(2, 1, 3))
model3 <- arima(y_ts, order = c(1, 1, 4))

print(model1)
print(model2)
print(model3)

coef_model1 <- coeftest(model1)
coef_model2 <- coeftest(model2)
coef_model3 <- coeftest(model3)

print(coef_model1)
print(coef_model2)
print(coef_model3)

# -----------------------------
# 3. Diagnostic tests
# -----------------------------

res1 <- residuals(model1)
res2 <- residuals(model2)
res3 <- residuals(model3)

# Ljung-Box tests
lb1 <- Box.test(res1, lag = 10, type = "Ljung-Box")
lb2 <- Box.test(res2, lag = 10, type = "Ljung-Box")
lb3 <- Box.test(res3, lag = 10, type = "Ljung-Box")

print(lb1)
print(lb2)
print(lb3)

# Residual ACF/PACF for selected model only
png("figures/residual_diagnostics.png", width = 1000, height = 500)
par(mfrow = c(1, 2))
acf(res1,  main = "ACF of Residuals: ARIMA(1,1,3)")
pacf(res1, main = "PACF of Residuals: ARIMA(1,1,3)")
dev.off()

# Normality diagnostics
shapiro_res1 <- shapiro.test(res1)
print(shapiro_res1)

png("figures/residual_normality.png", width = 1000, height = 500)
par(mfrow = c(1, 2))
hist(res1, main = "Histogram of Residuals", xlab = "Residuals")
qqnorm(res1, main = "Normal Q-Q Plot")
qqline(res1)
dev.off()

# -----------------------------
# 4. Fitted values and forecasting
# -----------------------------

# Original vs fitted
png("figures/original_vs_fitted.png", width = 900, height = 500)
plot(y_ts, type = "l",
     main = "Original Series and Fitted Values",
     ylab = "Y_t",
     xlab = "Time")
lines(fitted(model1), col = "red", lwd = 1.5)
legend("topleft", legend = c("Original", "Fitted"),
       col = c("black", "red"), lty = 1, bty = "n")
dev.off()

# Forecast horizons
forecast10  <- forecast(model1, h = 10)
forecast25  <- forecast(model1, h = 25)
forecast100 <- forecast(model1, h = 100)

# Forecast plot for h = 100 with actual test data
png("figures/forecast_vs_actual.png", width = 1000, height = 500)
plot(forecast100, main = "100-Step Ahead Forecast vs Actual")
lines(ts(y_test, start = length(y_train) + 1), col = "blue", lwd = 1.5)
legend("topleft", legend = c("Forecast", "Actual test series"),
       col = c("black", "blue"), lty = 1, bty = "n")
dev.off()

# MSE
mse <- mean((y_test - forecast100$mean)^2)
print(mse)

# -----------------------------
# 5. Summary objects
# -----------------------------

cat("\nSelected model: ARIMA(1,1,3)\n")
cat("ADF p-value (original series):", adf_original$p.value, "\n")
cat("ADF p-value (first difference):", adf_diff1$p.value, "\n")
cat("Ljung-Box p-value (selected model):", lb1$p.value, "\n")
cat("Shapiro-Wilk p-value (selected model):", shapiro_res1$p.value, "\n")
cat("Out-of-sample MSE:", mse, "\n")