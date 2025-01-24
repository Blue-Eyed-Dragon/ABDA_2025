# Load necessary libraries
library(brms)

# Load the preprocessed dataset
insurance_data <- read.csv("Encoded_Insurance_Fraud_Dataset_Retry.csv")

# Convert the target variable to a factor
insurance_data$fraud_reported <- as.factor(insurance_data$fraud_reported)

# =========================
# Build the Hierarchical Model
# =========================
hierarchical_model <- brm(
  formula = fraud_reported ~ total_claim_sum + incident_hour_of_the_day + policy_deductable + 
    months_as_customer + witnesses + collision_type_Rear.Collision + collision_type_Side.Collision + 
    property_damage + police_report_available + 
    incident_severity_Minor.Damage + incident_severity_Total.Loss + 
    incident_severity_Trivial.Damage + 
    insured_education_level + 
    (1 + total_claim_sum | incident_state),
  data = insurance_data,
  family = bernoulli(link = "logit"),
  backend = "cmdstanr",
  chains = 4,
  iter = 2000,
  warmup = 1000,
  cores = 4,
  seed = 123,
  prior = c(
    prior(normal(0, 0.25), class = "b"),                           # Strong priors for fixed effects
    prior(normal(-2, 0.5), class = "Intercept"),                    # Strong informative intercept
    prior(exponential(2), class = "sd", group = "incident_state"), # Prior for random intercept
    prior(normal(0, 0.25), class = "sd", coef = "total_claim_sum", group = "incident_state") # Prior for random slope
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
# Save the Model for Future Use
# =========================
saveRDS(hierarchical_model, file = "hierarchical_model.rds")
