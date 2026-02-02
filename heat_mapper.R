library(dplyr)
library(ggplot2)

data= read.csv("recent_results.csv")

data$suppress_cent = ifelse(data$suppress_cent == "True", 1, 0)
data$suppress_dece = ifelse(data$suppress_dece == "True", 1, 0)

# Select case
x_value = 0.5

# Filter data; can set desired conditions
slice <- data %>% 
  filter(x == x_value & R0 > 1.4 & R0 < 3.6 & kc < 10.5)

# heatmap
ggplot(slice, aes(x = kc, y = R0, fill = ratio)) +
  geom_tile() + 
  geom_point(
    data = subset(slice, suppress_cent == 1),
    colour = "white",
    size = 2
  ) +           
  scale_fill_viridis_c(option = "plasma") +
  labs(
    #title = paste("Heatmap of ratio for x =", x_value),
    x = "k/c",
    y = "R0",
    fill = "(scale)"
  ) +
  theme_minimal(base_size = 14)