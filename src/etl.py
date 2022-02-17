"""
Extract transform and load functions, which pull the game data off my Git, as well as historical PECOTA projections

To get the data read in and cleaned for you, call load_model_data() at the bottom of the script!
"""

import os
import random
import itertools
import datetime as dt
import numpy as np
import pandas as pd

from constants import (
	BP_TEAM_RENAMINGS,
	TEAM_ABBREV_MAPS,
	GAMEDATA_FILE,
	N_BATTER,
	N_PITCHER,
	FEATURES_BATTER_BYPLAYER,
	FEATURES_BATTER,
	FEATURES_PITCHER_BYPLAYER,
	FEATURES_PITCHER,
	FEATURES,
)


##########################################################################
# Utils to read in Baseball Prospectus' Public Projections (PECOTA)
##########################################################################


def parse_bp_df(bp_df):
	"""
	Parses a dataframe of Pecota Batting projections, which may need various
	column renamings and other standardizing operations.

	Args:
	  bp_df : pd.DataFrame
		A dataframe of Pecota projections.

	Returns : pd.DataFrame
	  A parsed dataframe of Pecota projections.
	"""
	### renaming rules, as columns differ season-to-season
	rename_dict = {
		"oba": "obp",
		"mlbid": "mlbam_id",
		"mlbcode": "mlbam_id",
		"firstname": "first_name",
		"lastname": "last_name",
	}
	### reduce columns
	bp_df.columns = [item.lower() for item in bp_df.columns]

	### rename and drop missing entries
	bp_df = (
		bp_df.rename(columns=rename_dict)
		.dropna(subset=["mlbam_id"])
		.fillna(
			value={
				"pa": 0.0,
				"ab": 0.0,
				"pa17": 0.0,
				"ab17": 0.0,
				"g": 0.0,
				"gs": 0.0,
				"ip": 0.0,
			}
		)
	)

	### if provided as a slash line, split up
	if "avg_obp_slg" in bp_df.columns:
		bp_df.loc[:, ["avg", "obp", "slg"]] = np.stack(
			[np.array(item.split("/")).astype("float32") for item in bp_df.avg_obp_slg]
		)
	### add in/concat name, if not already there
	if "name" not in bp_df.columns:
		bp_df["name"] = [
			i + " " + j for (i, j) in list(zip(bp_df.first_name, bp_df.last_name))
		]

	### fraa column changed configuration in '21
	if "fraa_str" in bp_df.columns:
		bp_df["fraa_val"] = bp_df["fraa"]
	### make sure team names are standardized
	bp_df["team"] = [
		BP_TEAM_RENAMINGS[item] if item in BP_TEAM_RENAMINGS.keys() else item
		for item in bp_df.team
	]
	return bp_df


def retrieve_pecota_pitcher():
	"""
	Reads in raw Pecota batter projections (CSV format) from IKM's Github.

	Returns : pd.DataFrame
	  A parsed dataframe of pecota projections, 2016-2021.
	"""
	result = []
	for season in range(2016, 2022):
		print(f"...reading season: {season}...")
		### setup URL of CSV
		raw_file_url = f"https://raw.githubusercontent.com/isaackleislemurphy/pythagoras/main/data/bp/pecota_pitcher_{season}.csv"
		### read file
		try:
			df_temp = pd.read_csv(raw_file_url)
		except UnicodeDecodeError:
			df_temp = pd.read_csv(raw_file_url, encoding="unicode_escape")
		df_temp = parse_bp_df(df_temp)
		### add season as column
		df_temp["season"] = [season] * df_temp.shape[0]
		### add to full list
		result.append(df_temp)
	result = pd.concat(result, axis=0).reset_index(drop=True)
	### make GR column
	result["gr"] = result["g"].values - result["gs"].values
	# scale up 2020, which is predicted at 160 games
	result.loc[result.season == 2020, ["gr", "gs", "qs", "warp"]] = (
		162 / 60
	) * result.loc[result.season == 2020, ["gr", "gs", "qs", "warp"]]
	### frustratingly, BP does not do K% or BB%, so K9 and BB9 must suffice
	result["bb_ip"] = np.divide(result.bb.values, result.ip.values + 1e-10)
	result["so_ip"] = np.divide(result.so.values, result.ip.values + 1e-10)
	return result[
		[
			"season",
			"team",
			"name",
			"mlbam_id",
			"gs",
			"gr",
			"qs",
			"era",
			"dra",
			"bb_ip",
			"so_ip",
			"warp",
		]
	]


def retrieve_pecota_batter():
	"""
	Reads in raw Pecota batter projections (CSV format) from IKM's Github.

	Returns : pd.DataFrame
	  A parsed dataframe of pecota projections, 2016-2021.
	"""
	result = []
	for season in range(2016, 2022):
		print(f"...reading season: {season}...")
		### setup URL of CSV
		raw_file_url = f"https://raw.githubusercontent.com/isaackleislemurphy/pythagoras/main/data/bp/pecota_batter_{season}.csv"
		### read file
		try:
			df_temp = pd.read_csv(raw_file_url)
		except UnicodeDecodeError:
			df_temp = pd.read_csv(raw_file_url, encoding="unicode_escape")
		df_temp = parse_bp_df(df_temp)
		### add season as column
		df_temp["season"] = [season] * df_temp.shape[0]
		### add to full list
		result.append(df_temp)
	result = pd.concat(result, axis=0).reset_index(drop=True)
	### make columns ratewise + make OPS
	result["ops"] = result.obp.values + result.slg.values
	result["so_pct"] = np.divide(result.so.values, result.pa.values)
	result["vorp_pa"] = np.divide(result.vorp.values, result.pa.values)
	result["warp_pa"] = np.divide(result.warp.values, result.pa.values)
	return result[
		[
			"season",
			"pos",
			"team",
			"name",
			"mlbam_id",
			"pa",
			"avg",
			"obp",
			"ops",
			"slg",
			"babip",
			"vorp",
			"warp",
			"vorp_pa",
			"warp_pa",
			"fraa_val",
			"sb",
			"cs",
			"so_pct",
		]
	]


##########################################################################
# Utils to read in MLB scores, as scraped with the help of Bill Petti
# https://github.com/BillPetti/baseballr/blob/master/R/get_game_pks_mlb.R
##########################################################################


def retrieve_game_data(filepath=GAMEDATA_FILE):
	"""
	Reads in scraped game results, also stored on IKM Github.

	Args:
	  filepath : str
		The Github CSV filepath for the scraped game data

	Returns : pd.DataFrame
	  A dataframe of game data

	"""

	game_data = (
		pd.read_csv(filepath)
		.iloc[:, 1:]
		.rename(
			columns={
				"teams.away.score": "away_score",
				"teams.home.score": "home_score",
				"teams.away.leagueRecord.wins": "away_w",
				"teams.away.leagueRecord.losses": "away_l",
				"teams.home.leagueRecord.wins": "home_w",
				"teams.home.leagueRecord.losses": "home_l",
				"officialDate": "date",
			}
		)
	)

	game_data["away_team"] = [
		TEAM_ABBREV_MAPS[item] for item in game_data["teams.away.team.name"]
	]
	game_data["home_team"] = [
		TEAM_ABBREV_MAPS[item] for item in game_data["teams.home.team.name"]
	]

	### watch out for rain delays!
	game_data = (
		game_data[
			[
				"season",
				"game_pk",
				"date",
				"away_score",
				"home_score",
				"away_team",
				"home_team",
				"away_w",
				"away_l",
				"home_w",
				"home_l",
			]
		]
		.sort_values(["game_pk", "date"])
		.groupby(["game_pk"], as_index=False)
		.tail(1)
	)
	return game_data


#########################################################################
## Wrangling/Preprocessing Functions
#########################################################################


def widen_projection(projections_df, features, n_keep, rank_criteria):
	"""
	Given each team's projection, widens the projection df by team so that the first
	set of columns correspond to a team's best batter, the second set to a team's second best batter, ...,
	all the way through n_keep batter. "Best" is determined by rank_criteria.

	Args:
			projections_df : pd.DataFrame
					A pandas df of parsed projections, via functions above
			features : list[str]
					features to keep by player, in parsed projection_df
			n_keep : int
					Number of players to keep as "top-N" players
			rank_criteria : str
					Column in projections_df to sort player ability by

	Returns : pd.DataFrame
			A pandas df of widened projections.
	"""
	result = []
	for grp, df in projections_df.query("team != 'orphan'").groupby(
		["team", "season"], as_index=False
	):
		x_temp = (
			df.sort_values([rank_criteria], ascending=False)
			.head(n_keep)[features]
			.values.flatten()
		)
		df_temp = pd.DataFrame(x_temp).T
		df_temp.columns = [
			f"{j}_{i}" for i, j in itertools.product(range(1, n_keep + 1), features)
		]
		### save for rearrangement
		stat_cols = df_temp.columns.to_list()
		### add in groups as columns
		df_temp["team"] = grp[0]
		df_temp["season"] = grp[1]
		result.append(df_temp[["team", "season"] + stat_cols])
	result = pd.concat(result, axis=0).reset_index(drop=True)
	return result


def pivot_game_results(game_results):
	"""
	Given raw dataframe of game results, pivots them to long format
	for joining onto design matrix.
	"""
	### elongate df
	game_results_long = (
		pd.concat(
			[
				game_results.rename(
					columns={
						"away_score": "runs",
						"away_team": "batting_team",
						"home_team": "pitching_team",
					}
				)[
					[
						"season",
						"game_pk",
						"date",
						"runs",
						"batting_team",
						"pitching_team",
					]
				],
				game_results.rename(
					columns={
						"home_score": "runs",
						"home_team": "batting_team",
						"away_team": "pitching_team",
					}
				)[
					[
						"season",
						"game_pk",
						"date",
						"runs",
						"batting_team",
						"pitching_team",
					]
				],
			],
			axis=0,
		)
		.sort_values(["season", "game_pk"])
		.reset_index(drop=True)
	)

	return game_results_long


def wrangle_inseason_stats(game_results_long):
	"""
	Computes in-season run totals, runs allowed, and run differentials by team and season.

	Args:
	  game_results_long : pd.DataFrame
		A "long" game results dataframe, as outputted by pivot_game_results().

	Returns : pd.DataFrame
	  A pandas dataframe with in-season run totals.
	"""

	### temporal sort
	### offense
	off_prowess = (
		game_results_long.copy()
		.sort_values(["season", "batting_team", "date", "game_pk"])
		.reset_index(drop=True)
		.rename(columns={"batting_team": "team"})
		.drop(columns=["pitching_team"])
	)
	### defense
	def_prowess = (
		game_results_long.copy()
		.sort_values(["season", "pitching_team", "date", "game_pk"])
		.reset_index(drop=True)
		.rename(columns={"pitching_team": "team"})
		.drop(columns=["batting_team"])
	)

	### running totals for runs scored
	off_prowess[["runs_scored"]] = (
		off_prowess.groupby(["season", "team"])["runs"].cumsum().values
		- off_prowess.runs.values
	)

	### running totals for runs allowed
	def_prowess["runs_surrendered"] = (
		def_prowess.groupby(["season", "team"])["runs"].cumsum().values
		- def_prowess.runs.values
	)

	### put it all together
	total_prowess = pd.merge(
		off_prowess, def_prowess, on=["season", "game_pk", "date", "team"]
	).sort_values(["season", "team", "date", "game_pk"])

	### add in game number
	total_prowess["game_number"] = [1.0] * total_prowess.shape[0]
	total_prowess["game_number"] = (
		total_prowess.groupby(["season", "team"])["game_number"].cumsum().values
	)

	### run diff
	total_prowess["run_diff"] = np.subtract(
		total_prowess["runs_scored"].values, total_prowess["runs_surrendered"].values
	)

	### rates
	total_prowess["r_per_game"] = np.divide(
		total_prowess["runs_scored"].values, total_prowess["game_number"].values - 1.0
	)

	total_prowess["ra_per_game"] = np.divide(
		total_prowess["runs_surrendered"].values,
		total_prowess["game_number"].values - 1.0,
	)

	total_prowess["rdiff_per_game"] = np.divide(
		total_prowess["run_diff"].values, total_prowess["game_number"].values - 1.0
	)

	# fill game one nans
	total_prowess.fillna(
		value={"r_per_game": 0.0, "ra_per_game": 0.0, "rdiff_per_game": 0.0},
		inplace=True,
	)

	return total_prowess


def load_model_data():
	"""
	Loads all of the necessary projection + game data, fully processed.

	Returns : pd.DataFrame
	"""
	### read in raw data
	projections_batter = retrieve_pecota_batter()
	projections_pitcher = retrieve_pecota_pitcher()
	game_results = retrieve_game_data()

	### read in batter projections
	proj_batter_wide = widen_projection(
		projections_batter.copy(),
		features=FEATURES_BATTER_BYPLAYER,
		n_keep=N_BATTER,
		rank_criteria="warp_pa",
	)

	### read in pitcher projections
	proj_pitcher_wide = widen_projection(
		projections_pitcher.copy(),
		features=FEATURES_PITCHER_BYPLAYER,
		n_keep=N_PITCHER,
		rank_criteria="warp",
	)

	### wrangling
	game_results_long = pivot_game_results(game_results)

	model_data_df = game_results_long.merge(
		proj_batter_wide.rename(columns={"team": "batting_team"}),
		how="inner",
		on=["season", "batting_team"],
	).merge(
		proj_pitcher_wide.rename(columns={"team": "pitching_team"}),
		how="inner",
		on=["season", "pitching_team"],
		suffixes=["_bat", "_pitch"],
	)
	return model_data_df


if __name__ == '__main__':
	main()
