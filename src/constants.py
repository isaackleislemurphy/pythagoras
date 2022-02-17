"""Constants or Pecota Weibull prediction"""
import itertools
### BP's team abbreviations change year-on-year,
### so this dict helps standardize
BP_TEAM_RENAMINGS = {
	"ANA": "LAA",
	"CHA": "CHW",
	"CHN": "CHC",
	"KCA": "KC",
	"LAN": "LAD",
	"NYA": "NYY",
	"NYN": "NYM",
	"SDN": "SD",
	"SFN": "SF",
	"SLN": "STL",
	"TBA": "TB",
}

### converts a team's full name, as listed by MLB
### to it's three letter abbreviation
TEAM_ABBREV_MAPS = {
	"Arizona Diamondbacks": "ARI",
	"Atlanta Braves": "ATL",
	"Baltimore Orioles": "BAL",
	"Boston Red Sox": "BOS",
	"Chicago Cubs": "CHC",
	"Chicago White Sox": "CHW",
	"Cincinnati Reds": "CIN",
	"Cleveland Indians": "CLE",
	"Colorado Rockies": "COL",
	"Detroit Tigers": "DET",
	"Houston Astros": "HOU",
	"Kansas City Royals": "KC",
	"Los Angeles Angels": "LAA",
	"Los Angeles Dodgers": "LAD",
	"Miami Marlins": "MIA",
	"Milwaukee Brewers": "MIL",
	"Minnesota Twins": "MIN",
	"New York Mets": "NYM",
	"New York Yankees": "NYY",
	"Oakland Athletics": "OAK",
	"Philadelphia Phillies": "PHI",
	"Pittsburgh Pirates": "PIT",
	"San Diego Padres": "SD",
	"San Francisco Giants": "SF",
	"Seattle Mariners": "SEA",
	"St. Louis Cardinals": "STL",
	"Tampa Bay Rays": "TB",
	"Texas Rangers": "TEX",
	"Toronto Blue Jays": "TOR",
	"Washington Nationals": "WAS",
}

### file with game data
GAMEDATA_FILE = "https://raw.githubusercontent.com/isaackleislemurphy/pythagoras/main/data/game_df_petti.csv"

### model features and such
N_BATTER = 15  # take the stats of the top N projected batters by each team
N_PITCHER = 20  # same for pitchers
### batting features
FEATURES_BATTER_BYPLAYER = [
	# "avg",
	"obp",
	# "slg",
	# "babip",
	"vorp_pa",
	"warp_pa",
	# "so_pct"
]
FEATURES_BATTER = [
	f"{j}_{i}"
	for i, j in itertools.product(range(1, N_BATTER + 1), FEATURES_BATTER_BYPLAYER)
]
### pitching features
FEATURES_PITCHER_BYPLAYER = [
	"gr",
	"gs",
	# "era",
	"dra",
	"warp",
	# "so_ip",
	# "bb_ip"
]
FEATURES_PITCHER = [
	f"{j}_{i}"
	for i, j in itertools.product(range(1, N_PITCHER + 1), FEATURES_PITCHER_BYPLAYER)
]

FEATURES = FEATURES_BATTER + FEATURES_PITCHER
