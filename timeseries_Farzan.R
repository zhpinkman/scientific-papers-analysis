

# Load required libraries
library(tidyverse)
library(nlme)        # For GLS models
library(broom)       # For model summaries
library(ggplot2)     # Visualization

# =====================
# Step 1: Load the Data
# =====================
data <- read_csv("reddit_pca.csv")  # Replace with your file location

# View data structure
head(data)

# =========================
# Step 2: Data Preparation
# =========================
data <- data %>%
  mutate(
    time = as.Date(paste(year, month, "01", sep = "-")),
    ONSET = ifelse(time >= as.Date("2022-11-01"), 1, 0),
    POST = ifelse(ONSET == 1, as.numeric(time - as.Date("2022-11-01")), 0)
  )

# Verify data after preparation
head(data)

# ===============================
# Step 3: Fit GLS for Each Column
# ===============================

# Function to fit GLS for a given column
fit_gls <- function(column_name) {
  formula <- as.formula(paste(column_name, "~ time + ONSET + POST"))
  
  # Fit GLS model with AR(1) correlation structure
  model <- gls(
    formula, 
    data = data, 
    correlation = corAR1(form = ~ as.numeric(time))
  )
  
  return(model)
}

# Select only PC columns
pc_columns <- names(data)[grepl("^PC", names(data))]

# Fit GLS models
gls_models <- map(pc_columns, fit_gls)

# =======================================
# Step 4: Summarize and Filter GLS Results
# =======================================

# Function to extract results from GLS models
extract_gls_results <- function(model, column_name) {
  summary_model <- summary(model)
  coefficients <- summary_model$tTable
  
  # Convert coefficients to a data frame
  results_df <- as.data.frame(coefficients) %>%
    rownames_to_column(var = "term") %>%
    mutate(
      PC = column_name,
      estimate = Value,
      std.error = Std.Error,
      statistic = `t-value`,
      p.value = `p-value`
    ) %>%
    select(PC, term, estimate, std.error, statistic, p.value)
  
  return(results_df)
}

# Extract results for each GLS model
gls_results <- map_df(
  seq_along(pc_columns),
  ~extract_gls_results(gls_models[[.x]], pc_columns[.x])
)

# Filter for significant results
significant_gls_results <- gls_results %>%
  filter(p.value < 0.05)

# Print significant results
print(significant_gls_results)

# ==========================
# Step 5: Assumption Checks
# ==========================
# Function to check residual autocorrelation
check_residuals <- function(model, column_name) {
  residuals <- residuals(model, type = "normalized")
  
  # Perform Durbin-Watson Test
  dw_test <- nlme::corAR1(form = ~ as.numeric(time))
  
  cat("### Assumption Checks for:", column_name, "###\n")
  print(summary(dw_test))
}

# Run assumption checks for significant models
significant_pcs <- unique(significant_gls_results$PC)
significant_gls_models <- gls_models[pc_columns %in% significant_pcs]

map2(
  significant_gls_models,
  significant_pcs,
  ~check_residuals(.x, .y)
)

# ==========================
# Step 6: Visualization
# ==========================
visualize_gls <- function(column_name, model) {
  ggplot(data, aes(x = time, y = .data[[column_name]])) +
    geom_point(alpha = 0.5) +
    geom_smooth(method = "lm", color = "blue") +
    labs(
      title = paste("GLS Model for", column_name),
      x = "Time",
      y = column_name
    ) +
    theme_minimal()
}

# Visualize significant results
for (pc in significant_pcs) {
  model <- gls_models[[which(pc_columns == pc)]]
  print(visualize_gls(pc, model))
}

# ===========================
# Step 7: Export Results
# ===========================
write_csv(significant_gls_results, "significant_results_reddit.csv")