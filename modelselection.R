library(tidyverse)
library(rsample)
library(randomForest)
library(gbm)


shots <- read.csv("data/pbpstats-tracking-shots.csv")
players <- read.csv("data/player_names.csv")

players <- players %>%
  dplyr::select(-1)


shots <- merge(shots, players, by.x = "player_id", by.y = "id")



shots$close_def_dist <- factor(shots$close_def_dist, 
                               levels = c("0-2 Feet - Very Tight", 
                                          "2-4 Feet - Tight",
                                          "4-6 Feet - Open",
                                          "6+ Feet - Wide Open"))
shots$shot_clock <- factor(shots$shot_clock, 
                           levels = c("4-0 Very Late", 
                                      "7-4 Late",
                                      "15-7 Average",
                                      "18-15 Early",
                                      "22-18 Very Early"))
shots <- shots %>%
  mutate(shot_clock = replace_na(shot_clock, "15-7 Average"))
shots$touch_time <- factor(shots$touch_time, 
                           levels = c("Touch < 2 Seconds", 
                                      "Touch 2-6 Seconds",
                                      "Touch 6+ Seconds"))
shots$dribble_range <- factor(shots$dribble_range, 
                              levels = c("0 Dribbles", 
                                         "1 Dribble",
                                         "2 Dribbles",
                                         "3-6 Dribbles",
                                         "7+ Dribbles"))
shots <- shots %>% 
  mutate(points = if_else(u10_ft_fg2m == 1, 2, 
                          if_else(u10_ft_fg2m == 1, 2,
                                  if_else(fg3m == 1, 3, 0))))





shot_split <- shots %>%
  initial_split(prop = 0.75)
# fit <- glm(fg3m ~ close_def_dist*shot_clock*touch_time*dribble_range, data = shots3)
# step(fit, direction = "both")

train_shots <- training(shot_split)
test_shots <- testing(shot_split)

fit1 <- glm(points ~ close_def_dist + shot_clock + dribble_range + 
              touch_time + u10_ft_fg2a + u10_ft_fg2a + 
              fg3a, data = train_shots)
summary(fit1)
predict(fit1, newdata = test_shots)

lgam <- gam(points ~ close_def_dist + shot_clock + touch_time + 
              dribble_range + s(u10_ft_fg2a, k = 8) + s(u10_ft_fg2a, k = 8) + 
              s(fg3a, k = 8), method = "ML", data = train_shots) 
summary(lgam)

shots1 <- shots %>%
  filter(game_id == 22300005)
pred.lgam <- predict(lgam, newdata = shots1)
sum(pred.lgam)
sum(shots1$points)

pred.fit <- predict(fit1, newdata = shots1)
sum(pred.fit)
sum(shots1$points)

rf <- randomForest(points ~ close_def_dist + shot_clock + dribble_range + 
                     touch_time + u10_ft_fg2a + u10_ft_fg2a + 
                     fg3a, data = train_shots, na.action = na.roughfix, 
                   importance = TRUE, ntree = 1000) 
print(rf)
install.packages("gbm")

gbmfit <- gbm(
  formula = points ~ close_def_dist + shot_clock + dribble_range + 
    touch_time + u10_ft_fg2a + u10_ft_fg2a + 
    fg3a, 
  data = train_shots,
  distribution = "gaussian",  # SSE loss function
  n.trees = 5000,
  shrinkage = 0.1,
  interaction.depth = 3,
  n.minobsinnode = 10,
  cv.folds = 10
)

best <- which.min(gbmfit$cv.error)
sqrt(gbmfit$cv.error[best])
gbm.perf(gbmfit, method = "cv")


hyper_grid <- expand.grid(
  learning_rate = c(0.3, 0.1, 0.05, 0.01, 0.005),
  RMSE = NA,
  trees = NA,
  time = NA
)

# execute grid search
for(i in seq_len(nrow(hyper_grid))) {
  
  # fit gbm
  set.seed(123)  # for reproducibility
  train_time <- system.time({
    m <- gbm(
      formula = points ~ close_def_dist + shot_clock + dribble_range + 
        touch_time + u10_ft_fg2a + u10_ft_fg2a + 
        fg3a, 
      data = train_shots,
      distribution = "gaussian",
      n.trees = 5000, 
      shrinkage = hyper_grid$learning_rate[i], 
      interaction.depth = 3, 
      n.minobsinnode = 10,
      cv.folds = 10 
    )
  })
  
  # add SSE, trees, and training time to results
  hyper_grid$RMSE[i]  <- sqrt(min(m$cv.error))
  hyper_grid$trees[i] <- which.min(m$cv.error)
  hyper_grid$Time[i]  <- train_time[["elapsed"]]
  
}

# results
arrange(hyper_grid, RMSE)