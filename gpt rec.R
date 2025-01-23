# Load necessary libraries
library(brms)
library(caret)

# Load the cleaned dataset
insurance_data <- read.csv("Processed_Insurance_Fraud_Dataset.csv")

# Convert target variable and group variable to factors
insurance_data$fraud_reported <- as.factor(insurance_data$fraud_reported)
insurance_data$incident_state <- as.factor(insurance_data$incident_state)

# Split the dataset into training and testing sets
set.seed(123)  # For reproducibility
train_indices <- sample(seq_len(nrow(insurance_data)), size = 0.6 * nrow(insurance_data))
train_data <- insurance_data[train_indices, ]
test_data <- insurance_data[-train_indices, ]

# =========================
# Build the Base Model
# =========================
base_model <- brm(
  formula = fraud_reported ~ property_damage + bodily_injuries + police_report_available + 
    total_claim_sum + incident_hour_of_the_day,
  data = train_data,
  family = bernoulli(link = "logit"),
  backend = "cmdstanr",
  chains = 4,
  iter = 2000,
  warmup = 1000,
  cores = 4,
  seed = 123,
  prior = c(
    prior(normal(0, 1), class = "b"),               # General fixed effects
    prior(normal(-2, 1), class = "Intercept"),      # Informative intercept
    prior(normal(0, 0.5), class = "b", coef = "property_damage"),
    prior(normal(0, 0.5), class = "b", coef = "police_report_available"),
    prior(normal(0, 2), class = "b", coef = "total_claim_sum")
  ),
  control = list(
    max_treedepth = 15,
    adapt_delta = 0.95
  )
)

# Summarize the Base Model
summary(base_model)

# Posterior Predictive Checks for Base Model
pp_check(base_model)

# =========================
# Build the Hierarchical Model
# =========================
hierarchical_model <- brm(
  formula = fraud_reported ~ property_damage + bodily_injuries + police_report_available + 
    total_claim_sum + incident_hour_of_the_day + number_of_vehicles_involved + 
    umbrella_limit + policy_deductable + (1 | incident_state),
  data = train_data,
  family = bernoulli(link = "logit"),
  backend = "cmdstanr",
  chains = 4,
  iter = 2000,
  warmup = 1000,
  cores = 4,
  seed = 123,
  prior = c(
    prior(normal(0, 0.5), class = "b"),               # Fixed effects
    prior(normal(-2, 1), class = "Intercept"),        # Informative intercept
    prior(exponential(1), class = "sd", group = "incident_state")  # Random effects
  ),
  control = list(
    max_treedepth = 15,
    adapt_delta = 0.95
  )
)

# Summarize the Hierarchical Model
summary(hierarchical_model)

# Posterior Predictive Checks for Hierarchical Model
pp_check(hierarchical_model)

# =========================
# Compare Models
# =========================
# Compare the base and hierarchical models using WAIC and LOO
waic_base <- waic(base_model)
waic_hierarchical <- waic(hierarchical_model)

loo_base <- loo(base_model)
loo_hierarchical <- loo(hierarchical_model)

# Print model comparison metrics
print(waic_base)
print(waic_hierarchical)
print(loo_base)
print(loo_hierarchical)

# =========================
# Predictions on Test Data
# =========================
# Predictions for the base model
predicted_probs_base <- posterior_predict(base_model, newdata = test_data, re.form = NA)
test_data$predicted_prob_base <- apply(predicted_probs_base, 2, mean)
test_data$predicted_class_base <- ifelse(test_data$predicted_prob_base > 0.5, 1, 0)

# Predictions for the hierarchical model
predicted_probs_hierarchical <- posterior_predict(hierarchical_model, newdata = test_data, re.form = ~ (1 | incident_state))
test_data$predicted_prob_hierarchical <- apply(predicted_probs_hierarchical, 2, mean)
test_data$predicted_class_hierarchical <- ifelse(test_data$predicted_prob_hierarchical > 0.5, 1, 0)

# =========================
# Evaluate Model Performance
# =========================
# Confusion Matrix for Base Model
confusion_base <- confusionMatrix(
  as.factor(test_data$predicted_class_base),
  as.factor(test_data$fraud_reported)
)
print(confusion_base)

# Confusion Matrix for Hierarchical Model
confusion_hierarchical <- confusionMatrix(
  as.factor(test_data$predicted_class_hierarchical),
  as.factor(test_data$fraud_reported)
)
print(confusion_hierarchical)
