# Load necessary libraries
library(brms)
library(caret)

# Load the dataset
# Assuming the dataset is saved as 'insurance_data_encoded.csv'
insurance_data <- read.csv("insurance_data_encoded.csv")

# Convert the target variable (fraud_reported) to a factor
insurance_data$fraud_reported <- as.factor(insurance_data$fraud_reported)

# Split the dataset into training and testing sets
set.seed(123)  # For reproducibility
train_indices <- sample(seq_len(nrow(insurance_data)), size = 0.6 * nrow(insurance_data))
train_data <- insurance_data[train_indices, ]
test_data <- insurance_data[-train_indices, ]

# Define the Bayesian Logistic Regression model
# Use predictors from the dataset (modify the formula as needed)
bayesian_model <- brm(
  formula = fraud_reported ~ age + policy_deductable + property_damage + bodily_injuries + police_report_available + total_claim_amount,
  data = train_data,
  family = bernoulli(link = "logit"),
  backend = "cmdstanr",
  chains = 4,  # Number of MCMC chains
  iter = 2000,  # Total iterations per chain
  warmup = 1000,  # Warm-up iterations per chain
  cores = 4,  # Use all available cores for faster computation
  seed = 123,  # For reproducibility
  prior = c(
    prior(normal(0, 1), class = "b"),             # Priors for most coefficients
    prior(normal(-2.2, 1), class = "Intercept"),  # Prior for the intercept
    prior(normal(0, 0.5), class = "b", coef = "property_damage"),  # Regularizing prior
    prior(normal(0, 0.5), class = "b", coef = "police_report_available"),  # Regularizing prior
    prior(normal(0, 2), class = "b", coef = "total_claim_amount")  # Wide prior for scaled variable
  ),
  control = list(
    max_treedepth = 15,
    adapt_delta = 0.95
  )
)

# Summary of the model
summary(bayesian_model)

# Posterior predictive checks
pp_check(bayesian_model)

# Model Diagnostics: Trace plots and convergence
plot(bayesian_model)

# Generate predictions on the test set
predicted_probs <- posterior_predict(bayesian_model, newdata = test_data, re.form = NA)

# Aggregate predictions by averaging across posterior draws (rows of the matrix)
test_data$predicted_prob <- apply(predicted_probs, 2, mean)  # Column-wise mean

# Convert predicted probabilities to binary outcomes
test_data$predicted_class <- ifelse(test_data$predicted_prob > 0.5, 1, 0)

# Evaluate model performance using a confusion matrix
confusion_matrix <- confusionMatrix(
  as.factor(test_data$predicted_class), 
  as.factor(test_data$fraud_reported)
)

# Print the confusion matrix
print(confusion_matrix)
