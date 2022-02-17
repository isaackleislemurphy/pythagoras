### setup
library(dplyr)
library(ggplot2)
library(lubridate)

### read in data
game_df = read.csv("~/pythagoras/data/game_df_petti.csv")

### bit of data munging TIL: the 2016 Cubs tied a game?!?
game_df = game_df %>%
  # convert to int
  mutate(teams.away.isWinner=as.numeric(teams.away.isWinner)) %>%
  mutate(teams.away.isWinner=ifelse(
    is.na(teams.away.isWinner), .5, teams.away.isWinner
  )) %>%
  arrange(season, teams.away.team.name, teams.home.team.name, game_pk, officialDate) %>%
  group_by(game_pk) %>%
  # BAM feed provied a couple of duplicates, so drop those
  summarise_all(., first) %>%
  ungroup()

collapse_win_loss <- function(game_df_in){
  #' Computes W/L/RS/RA for each team and season
  #' @param game_df_in: data.frame. A wrangled data frame
  #' @return data.frame. A dataframe containing the aforementioned. 
  ### collapse down each teams' W/L/RS/RA, by season
  win_loss_df = bind_rows(
    # as an away team
    game_df_in %>%
      group_by(season, teams.away.team.name) %>%
      summarise(
        W=sum(teams.away.isWinner),
        L=sum(1 - teams.away.isWinner),
        RS=sum(teams.away.score),
        RA=sum(teams.home.score),
      ) %>%
      rename(team=teams.away.team.name)
    ,
    game_df_in %>%
      group_by(season, teams.home.team.name) %>%
      summarise(
        W=sum(1 - teams.away.isWinner),
        L=sum(teams.away.isWinner),
        RS=sum(teams.home.score),
        RA=sum(teams.away.score),
      ) %>%
      rename(team=teams.home.team.name)
  ) %>%
    group_by(season, team) %>%
    summarise_all(., sum) %>%
    mutate(
      G=W+L,
      win_pct=W/G,
      pythag_win_pct=(RS^1.82 / (RS^1.82 + RA^1.82)),
      fortune=win_pct - pythag_win_pct
    )
  win_loss_df
}




# Explanatory -------------------------------------------------------------

### plot the differences between Pythag and Observed
# collapse
win_loss_df = collapse_win_loss(game_df)
# plot
ggplot(
  win_loss_df,
  aes(
    x=pythag_win_pct,
    y=win_pct,
  )
) +
  geom_point() +
  facet_wrap(~as.factor(season)) +
  labs(
    title="Pythagorean Win% vs. Observed (2016-2021)",
    subtitle="vs. Perfect Fit",
    x="Pythagorean Win % (Gamma=1.82)",
    y="Observed Win %"
  ) + 
  theme(
    plot.title = element_text(hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5)
  ) +
  geom_abline(slope=1, color="blue")


### top 5 luckiest teams over last 6 seasons
win_loss_df %>%
  arrange(desc(fortune)) %>%
  head(5)


### top 5 unluckiest teams over the last 6 seasons
win_loss_df %>%
  arrange(fortune) %>%
  head(5)


### deep dive on 2021 Mariners
# suppose their "true" win probability entering each game
# was their pythagorean runs. We can then use this as the rate
# parameter of a binomial, and compute the CDF under this "true"
# win probability of achieving the wins that they did
sea_2021 = win_loss_df %>%
  filter(team == 'Seattle Mariners', season == 2021)

# P(Wins >= X), X ~ binom(162, pythag win %)
1 - pbinom(sea_2021$W - 1, sea_2021$G, sea_2021$pythag_win_pct)



# Prediction --------------------------------------------------------------

# put pre-deadline and post-deadline side by side
wl_df_od_july = collapse_win_loss(
  game_df %>% filter(season != 2020, month(officialDate) <= 7)
)
wl_df_aug_sep = collapse_win_loss(
  game_df %>% filter(season != 2020, month(officialDate) >= 8)
)

wl_df_split_season = inner_join(
  wl_df_od_july,
  wl_df_aug_sep,
  by=c("team", "season"),
  suffix=c("_od_july", "_aug_sep")
)

# plot
ggplot(
  wl_df_split_season,
  aes(
    x=pythag_win_pct_od_july,
    y=win_pct_aug_sep,
  )
) +
  geom_point() +
  geom_point(
    mapping=aes(
      x=win_pct_od_july,
      y=win_pct_aug_sep,
    ),
    color="blue"
  ) +
  facet_wrap(~as.factor(season)) +
  labs(
    title="Predicted Win % Aug-Sep",
    subtitle="Blue = Observed  || Black = Pythagorean || Red = Perfect Prediction",
    x="Win % OD-July (Blue)    ||    Pythag Win % OD-July (Black)",
    y="Observed Win % Aug-Sep"
  ) + 
  theme(
    plot.title = element_text(hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5)
  ) +
  geom_abline(slope=1, color="red")

### RMSE
# pythagorean
mean((wl_df_split_season$pythag_win_pct_od_july - wl_df_split_season$win_pct_aug_sep)^2)^(1/2)
# actual
mean((wl_df_split_season$win_pct_od_july - wl_df_split_season$win_pct_aug_sep)^2)^(1/2)

### MAE
# pythagorean
mean(abs(wl_df_split_season$pythag_win_pct_od_july - wl_df_split_season$win_pct_aug_sep))
# actual
mean(abs(wl_df_split_season$win_pct_od_july - wl_df_split_season$win_pct_aug_sep))






