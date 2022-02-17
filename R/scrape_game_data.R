#' The following code is lightly adapted from Bill Petti's "baseballr" package;
#' all credit to Petti here. 
suppressMessages(library(dplyr))
suppressMessages(library(lubridate))

get_game_pks_mlb <- function(date,
                             level_ids = c(1)) {
  #' Gets the MLBAM game_pk field for all MLB games on a particular date
  api_call <- paste0("http://statsapi.mlb.com/api/v1/schedule?sportId=", paste(level_ids, collapse = ','), "&date=", date)
  
  payload <- jsonlite::fromJSON(api_call, flatten = TRUE)
  
  payload <- payload$dates$games %>%
    as.data.frame() %>%
    rename(game_pk = .data$gamePk)
  
  return(payload)
  
}

### iterate over dates and pull data
lapply(
  seq(ymd("2016-03-01"), ymd("2021-10-10"), 1), function(d){
    print(d)
    result = try(get_game_pks_mlb(d))
    if(class(result) == "data.frame"){
      return(result)
    }else{
      return(data.frame())
    }
  }
  ) -> gamez

#$$ narrow down df
game_df = do.call("bind_rows", gamez) %>%
  filter(gameType == "R", status.codedGameState == "F") %>%
  dplyr::select(
    game_pk,
    season,
    teams.away.isWinner, 
    teams.away.team.name,
    teams.home.team.name,
    game_pk,
    teams.away.score,
    teams.home.score,
    teams.away.leagueRecord.wins,
    teams.away.leagueRecord.losses,
    teams.home.leagueRecord.wins,
    teams.home.leagueRecord.losses,
    everything()
    )

### write csv
game_df("./data/game_df_petti.csv")

