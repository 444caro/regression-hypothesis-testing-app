# Hypothesis Testing & Confidence Intervals in Linear Regression (Flask App)

This interactive web application demonstrates **hypothesis testing** and **confidence interval construction** in the context of simple linear regression. Built with Flask and Matplotlib, users can simulate datasets, visualize results, test regression assumptions, and construct confidence intervals using the t-distribution.

---

## Objective

To provide a visual and computational tool to understand:
- How sampling variability impacts linear regression estimates
- How p-values are computed using simulation-based hypothesis testing
- How confidence intervals are constructed and interpreted
- The effects of sample size, error variance, and true model parameters on inference

---

## Technologies Used

- **Python 3**
- **Flask** – backend and routing
- **Matplotlib** – server-side plot rendering
- **NumPy** – data generation and simulation
- **SciPy** – t-distribution for confidence intervals
- **Jinja2 Templates** – HTML rendering

---

## Features

### Linear Regression Simulation
- Simulate a dataset using the model: `Y = β₀ + β₁X + μ + ε`
- Plot the regression line fitted to the data

### Inference Through Repeated Simulations
- Run `S` simulations to collect slopes and intercepts
- Generate histograms of slope/intercept distributions

### Hypothesis Testing
- Compare observed statistic vs. null hypothesis (`H₀`)
- Choose one-sided or two-sided tests
- Visualize p-value region and simulated distribution

### Confidence Intervals
- Build t-distribution-based confidence intervals for slope/intercept
- Visualize interval coverage and placement
- Check if the interval includes the true parameter

---
