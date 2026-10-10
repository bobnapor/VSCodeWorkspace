import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from bs4 import Comment
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import resample
import logging
import concurrent.futures
import os
from datetime import datetime
import requests
import time

# URL templates for web scraping
games_url = 'https://www.pro-football-reference.com/years/yyyy/games.htm'
def_url = 'https://www.pro-football-reference.com/years/yyyy/opp.htm'
off_url = 'https://www.pro-football-reference.com/years/yyyy/'

# Optional: Keep file templates as fallback
file_dir = 'C:/Users/Bobby/Downloads/NFL_Stats/'
games_template = file_dir + 'yyyy NFL Regular Season Schedule _ Pro-Football-Reference.com.html'
def_template = file_dir + 'yyyy NFL Opposition & Defensive Statistics _ Pro-Football-Reference.com.html'
off_template = file_dir + 'yyyy NFL Standings & Team Stats _ Pro-Football-Reference.com.html'

# Global cache for web data (shared across iterations)
_web_data_cache = {}

# Number of past seasons loaded for training, in addition to the season being predicted
LOOKBACK_YEARS = 6

# Columns to use for stats
stat_columns_to_use = {
    'team',
    'pass_yds',
    'rush_yds',
    'turnovers'
}


def is_numeric_value(value):
    try:
        int(value)
        return True
    except ValueError:
        return False



# Directory where fetched HTML pages are cached locally to avoid re-fetching
HTML_CACHE_DIR = 'C:/Users/Bobby/Downloads/NFL_Stats_Cache/'


_web_blocked = False


class WebAccessBlocked(Exception):
    """Raised when the site refuses automated requests (HTTP 403/429 or a Cloudflare challenge)."""


def read_local_html(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()


def get_html(url, local_path, use_web=True):
    """Return page HTML from the web (if enabled), falling back to a saved local copy.

    pro-football-reference now blocks scripted requests, so pages saved from a
    browser into file_dir are used whenever the web fetch fails.
    """
    global _web_blocked
    if use_web and not _web_blocked:
        try:
            return fetch_url_with_retry(url)
        except requests.RequestException as e:
            logging.warning(f"Web fetch failed for {url} ({e}); trying local file {local_path}")
        except WebAccessBlocked as e:
            _web_blocked = True  # stop hitting the site for the remaining pages
            logging.warning(f"{e}; using local files from {file_dir} for the rest of this run")
    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"Could not fetch {url} and no local copy exists. "
            f"Open the page in a browser and save it as: {local_path}"
        )
    return read_local_html(local_path)


def fetch_url_with_retry(url, max_retries=3, delay=2):
    """Fetch URL content with retry logic, local file caching, and rate limiting.

    On first fetch, saves HTML to HTML_CACHE_DIR so subsequent runs never
    need to hit the site again for the same URL.
    """
    import hashlib
    import os

    # Build a stable cache filename from the URL
    os.makedirs(HTML_CACHE_DIR, exist_ok=True)
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache_path = os.path.join(HTML_CACHE_DIR, f"{url_hash}.html")

    # Return cached version if it exists
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()

    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        ),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.pro-football-reference.com/',
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            # pro-football-reference sits behind a Cloudflare challenge; retrying won't help
            if response.status_code in (403, 429) or response.headers.get('cf-mitigated') == 'challenge':
                raise WebAccessBlocked(f"{response.status_code} blocked by site protection for {url}")
            response.raise_for_status()
            html = response.text
            # Save to cache
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(html)
            return html
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))  # Exponential backoff
            else:
                raise e
    return None


def get_offense_stats(full_offense_soup, year_str):
    single_year_offense = []
    for comment in full_offense_soup.find_all(string=lambda text: isinstance(text, Comment)):
        offense_soup = BeautifulSoup(comment, 'html.parser')
        offense_table = offense_soup.find('table', id='team_stats')
        if offense_table:
            offense_rows = offense_table.find('tbody').find_all('tr')
            for offense_row in offense_rows:
                num_games = int(next((col.text for col in offense_row.find_all('td') if col['data-stat'] == 'g'), '0'))
                single_team_offense = {col['data-stat']: '0' if col.text.strip() == '' else str(int(col.text)/num_games) if is_numeric_value(col.text) else col.text for col in offense_row.find_all('td') if col['data-stat'] in stat_columns_to_use}
                team = single_team_offense.get('team')
                if team:
                    single_year_offense.append(single_team_offense)
    return pd.DataFrame(single_year_offense)


def get_defense_stats(full_defense_soup, year_str):
    single_year_defense = []
    defense_table = full_defense_soup.find('table', id='team_stats')
    if not defense_table:
        return pd.DataFrame()
    defense_rows = defense_table.find('tbody').find_all('tr')
    for defense_row in defense_rows:
        num_games = int(next((col.text for col in defense_row.find_all('td') if col['data-stat'] == 'g'), '0'))
        single_team_defense = {f'def_{col["data-stat"]}': '0' if col.text.strip() == '' else str(int(col.text)/num_games) if is_numeric_value(col.text) else col.text for col in defense_row.find_all('td') if col['data-stat'] in stat_columns_to_use}
        team = single_team_defense.get('def_team')
        if team:
            single_year_defense.append(single_team_defense)
    return pd.DataFrame(single_year_defense)


def get_single_year_team_stats(single_year_offense_df, single_year_defense_df, year_str):
    single_year_defense_df.rename(columns={'def_team':'team'}, inplace=True)
    single_year_combined_stats = pd.merge(single_year_offense_df, single_year_defense_df, on='team')
    single_year_combined_stats['year'] = year_str
    return single_year_combined_stats


def get_games(full_games_soup, year_str):
    games_table = full_games_soup.find('table', id='games')
    games_list_past = []
    games_list_future = []

    for game_row in games_table.find('tbody').find_all('tr'):
        this_game_week_num = game_row.find_all('th')[0].text.strip()
        game_stats = {col['data-stat']: col.text for col in game_row.find_all('td')}
        if not game_stats or 'game_date' not in game_stats or 'winner' not in game_stats or 'loser' not in game_stats or this_game_week_num == '':
            continue

        game_date = game_stats['game_date']
        away_team = game_stats['winner'] if game_stats.get('game_location') == '@' else game_stats['loser']
        home_team = game_stats['loser'] if away_team == game_stats['winner'] else game_stats['winner']

        if game_stats['pts_win'] != '' and game_stats['pts_lose'] != '':
            winner_score = int(game_stats['pts_win'])
            loser_score = int(game_stats['pts_lose'])
            score_difference = winner_score - loser_score if home_team == game_stats['winner'] else loser_score - winner_score

            games_list_past.append({
                'week_number': this_game_week_num,
                
                'game_date': game_date,
                'away_team': away_team,
                'home_team': home_team,
                'score_difference': score_difference,
                'year': year_str
            })
        else:
            games_list_future.append({
                'week_number': this_game_week_num,
                'game_date': game_date,
                'away_team': away_team,
                'home_team': home_team,
                'score_difference': 0,
                'year': year_str
            })

    return pd.DataFrame(games_list_past), pd.DataFrame(games_list_future)


def preprocess_data(football_data, games):
    merged_data = games.merge(
        football_data, how='left', left_on=['year', 'home_team'], right_on=['year', 'team']
    ).merge(
        football_data, how='left', left_on=['year', 'away_team'], right_on=['year', 'team'], suffixes=('_home', '_away')
    )

    features = [
        'pass_yds_home', 'def_pass_yds_home', 'rush_yds_home', 'def_rush_yds_home', 'turnovers_home', 'def_turnovers_home',
        'pass_yds_away', 'def_pass_yds_away', 'rush_yds_away', 'def_rush_yds_away', 'turnovers_away', 'def_turnovers_away'
    ]
    merged_data = merged_data.dropna(subset=features + ['score_difference'])

    return merged_data, features


def calculate_prediction_intervals(model, X, y, X_test, n_iterations=100, alpha=0.05):
    predictions = []
    for _ in range(n_iterations):
        X_resample, y_resample = resample(X, y)
        model.fit(X_resample, y_resample)
        predictions.append(model.predict(X_test))

    predictions = np.array(predictions)
    lower_bound = np.percentile(predictions, 100 * alpha / 2, axis=0)
    upper_bound = np.percentile(predictions, 100 * (1 - alpha / 2), axis=0)

    return lower_bound, upper_bound


def predict_game(home_team, away_team, year, football_data, model):
    home_stats = football_data[(football_data['team'] == home_team) & (football_data['year'] == year)]
    away_stats = football_data[(football_data['team'] == away_team) & (football_data['year'] == year)]

    if home_stats.empty or away_stats.empty:
        raise ValueError("Team statistics for the specified year not found.")

    input_features = pd.DataFrame([{
        'pass_yds_home': home_stats['pass_yds'].values[0],
        'def_pass_yds_home': home_stats['def_pass_yds'].values[0],
        'rush_yds_home': home_stats['rush_yds'].values[0],
        'def_rush_yds_home': home_stats['def_rush_yds'].values[0],
        'turnovers_home': home_stats['turnovers'].values[0],
        'def_turnovers_home': home_stats['def_turnovers'].values[0],
        'pass_yds_away': away_stats['pass_yds'].values[0],
        'def_pass_yds_away': away_stats['def_pass_yds'].values[0],
        'rush_yds_away': away_stats['rush_yds'].values[0],
        'def_rush_yds_away': away_stats['def_rush_yds'].values[0],
        'turnovers_away': away_stats['turnovers'].values[0],
        'def_turnovers_away': away_stats['def_turnovers'].values[0]
    }])

    predicted_score_difference = model.predict(input_features)[0]
    return predicted_score_difference


def load_data_for_year(year, off_url, def_url, games_url, use_web=True):
    """Load data for a year from web URLs (falling back to local files) or local files only"""
    year_str = str(year)

    off_html = get_html(off_url.replace('yyyy', year_str), off_template.replace('yyyy', year_str), use_web)
    single_year_offense = get_offense_stats(BeautifulSoup(off_html, 'html.parser'), year_str)

    def_html = get_html(def_url.replace('yyyy', year_str), def_template.replace('yyyy', year_str), use_web)
    single_year_defense = get_defense_stats(BeautifulSoup(def_html, 'html.parser'), year_str)

    single_year_combined_stats = get_single_year_team_stats(
        single_year_offense, single_year_defense, year_str
    )

    games_html = get_html(games_url.replace('yyyy', year_str), games_template.replace('yyyy', year_str), use_web)
    single_year_games_past, single_year_games_future = get_games(BeautifulSoup(games_html, 'html.parser'), year_str)

    return single_year_combined_stats, single_year_games_past, single_year_games_future

PROJECTION_HEADER = "Game|Week|Year|Home|Away|Mean|FairLine(Home)|StdDev|Actual"

# Std dev of (predicted - actual) home margin from nfl_prod_backtest.py over 2022-2025 regular seasons
# (1084 games, 200 runs per season, prior-season stats only so a game's result is never in its features).
# Used to turn a points edge into a cover probability.
NFL_MARGIN_STD_DEV = 13.9

SPREADSHEET_HEADERS = [
    'Week', 'Year', 'Home Team', 'Away Team',
    'Fair Line (as Home)', 'Spread (as Home)', 'Fair Line vs Spread',
    'Side to Cover', 'Actual Score Difference (as Home)', 'Covering Team', 'Result',
    'Cover Probability (Side to Cover)', 'Notes',
]
# Columns filled in by hand: Spread (F), Actual Score Difference (I), Notes (M)
MANUAL_INPUT_COLUMNS = (6, 9, 13)
MANUAL_INPUT_FILL_COLOR = 'FFFF99'
SPREADSHEET_COLUMN_WIDTHS = {
    'A': 30.3, 'B': 18.5, 'C': 22.7, 'D': 22.7, 'E': 18.6, 'F': 15.7,
    'G': 11.1, 'H': 22.7, 'I': 20.4, 'J': 22.7, 'K': 10.0,
    'L': 18.0, 'M': 21.7, 'O': 12.0,
}


def fair_line_as_home(projected_difference):
    """Projected home margin rounded to the nearest half point, same sign as the projection.

    +0.6 (home wins by 0.6) gives +0.5; -9.2 (home loses by 9.2) gives -9.0.
    """
    return round(projected_difference * 2) / 2 + 0.0


def get_desktop_path():
    """Desktop folder, honoring OneDrive redirection."""
    import ctypes
    from ctypes import wintypes
    buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
    CSIDL_DESKTOPDIRECTORY = 0x10
    ctypes.windll.shell32.SHGetFolderPathW(None, CSIDL_DESKTOPDIRECTORY, None, 0, buf)
    return buf.value


def _new_projections_workbook():
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    manual_fill = PatternFill(fill_type='solid', start_color=MANUAL_INPUT_FILL_COLOR, end_color=MANUAL_INPUT_FILL_COLOR)
    wb = Workbook()
    ws = wb.active
    ws.title = 'Sheet1'

    header_border = Border(bottom=Side(style='medium'))
    for col, title in enumerate(SPREADSHEET_HEADERS, start=1):
        cell = ws.cell(row=1, column=col, value=title)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell.border = header_border
    ws.row_dimensions[1].height = 29.5
    for letter, width in SPREADSHEET_COLUMN_WIDTHS.items():
        ws.column_dimensions[letter].width = width
    for col in MANUAL_INPUT_COLUMNS:
        ws.cell(1, col).fill = manual_fill

    ws['O1'] = 'Margin Std Dev'
    ws['O1'].font = Font(bold=True)
    ws['O1'].alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    ws['O2'] = NFL_MARGIN_STD_DEV
    return wb


def _load_or_create_projections_workbook(output_path):
    """Open the existing year workbook, or start a new one.

    An existing file whose header row doesn't match the current layout is copied to a timestamped
    backup instead of being appended to, so older layouts are never corrupted.
    """
    import shutil
    from openpyxl import load_workbook

    if not os.path.exists(output_path):
        return _new_projections_workbook()

    wb = load_workbook(output_path)
    ws = wb.active
    headers = [ws.cell(1, col).value for col in range(1, len(SPREADSHEET_HEADERS) + 1)]
    if headers == SPREADSHEET_HEADERS:
        return wb

    backup_path = output_path.replace('.xlsx', f'_old_layout_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
    shutil.copy2(output_path, backup_path)
    logging.warning(f"{output_path} has a different layout; copied it to {backup_path} and starting a new sheet")
    return _new_projections_workbook()


def write_projections_spreadsheet(rows, year, output_dir=None, actual_lookup=None):
    """Write projections to <Desktop>/<year>_nfl_projections.xlsx, appending to it if it already exists.

    The workbook acts as a running log for the season: each run adds its week's games in week order, so
    a week run after later weeks is inserted after the weeks before it and ahead of the weeks that follow.
    A game already in the sheet (same week, year and teams) keeps its original fair line, since
    that is the line that was bet on. Actual score differences are refreshed for every game in the sheet
    that has since been played, using `actual_lookup` ({(week, year, home, away): difference}) plus the
    rows passed in. Spread and notes are never touched. Layout follows the '2024 NFL Projections'
    workbook, with Notes moved to the far right; columns that need manual entry are highlighted and the
    rest are formulas.

    Returns (output_path, rows_added, actuals_updated).
    """
    from openpyxl.styles import PatternFill

    manual_fill = PatternFill(fill_type='solid', start_color=MANUAL_INPUT_FILL_COLOR, end_color=MANUAL_INPUT_FILL_COLOR)

    output_dir = output_dir or get_desktop_path()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{year}_nfl_projections.xlsx')

    wb = _load_or_create_projections_workbook(output_path)
    ws = wb.active

    def week_value(week):
        return int(week) if str(week).isdigit() else week

    def game_key(week, game_year, home, away):
        return (str(week_value(week)), str(game_year), home, away)

    def played_actual(actual):
        # Unplayed games carry a placeholder difference of 0
        return None if actual in (0, None) or pd.isna(actual) else int(actual)

    actuals = {}
    for key, actual in (actual_lookup or {}).items():
        if played_actual(actual) is not None:
            actuals[game_key(*key)] = played_actual(actual)
    for row in rows:
        if played_actual(row['actual']) is not None:
            actuals[game_key(row['week'], row['year'], row['home_team'], row['away_team'])] = played_actual(row['actual'])

    # Read the existing games so the sheet can be rewritten in week order
    records = []
    existing_keys = set()
    actuals_updated = 0
    for r in range(2, ws.max_row + 1):
        if not ws.cell(r, 3).value:
            continue
        record = {
            'week': ws.cell(r, 1).value, 'year': ws.cell(r, 2).value,
            'home': ws.cell(r, 3).value, 'away': ws.cell(r, 4).value,
            'fair_line': ws.cell(r, 5).value, 'spread': ws.cell(r, 6).value,
            'actual': ws.cell(r, 9).value, 'notes': ws.cell(r, 13).value,
        }
        key = game_key(record['week'], record['year'], record['home'], record['away'])
        existing_keys.add(key)
        if key in actuals and record['actual'] != actuals[key]:
            record['actual'] = actuals[key]
            actuals_updated += 1
        records.append(record)

    added = 0
    for row in rows:
        key = game_key(row['week'], row['year'], row['home_team'], row['away_team'])
        if key in existing_keys:
            continue
        existing_keys.add(key)
        records.append({
            'week': week_value(row['week']), 'year': int(row['year']),
            'home': row['home_team'], 'away': row['away_team'],
            'fair_line': fair_line_as_home(float(row['projected'])), 'spread': None,
            'actual': actuals.get(key), 'notes': None,
        })
        added += 1

    def season_order(record):
        week_number = translate_week_number(str(record['week']), record['year'])
        return (int(record['year']), week_number if week_number is not None else 99)

    # Stable sort: a week run late slots in after earlier weeks and before later ones, and games
    # within a week keep the order they were first written in
    records.sort(key=season_order)

    for r in range(2, ws.max_row + 1):
        for col in range(1, len(SPREADSHEET_HEADERS) + 1):
            ws.cell(r, col).value = None

    for r, record in enumerate(records, start=2):
        ws.cell(r, 1, record['week'])
        ws.cell(r, 2, record['year'])
        ws.cell(r, 3, record['home'])
        ws.cell(r, 4, record['away'])
        ws.cell(r, 5, record['fair_line']).number_format = '0.0'
        # F (spread), I (actual) and M (notes) are entered manually
        ws.cell(r, 6, record['spread'])
        ws.cell(r, 7, f'=E{r}+F{r}')
        ws.cell(r, 8, f'=IF(G{r}>0,C{r},D{r})')
        ws.cell(r, 9, record['actual'])
        ws.cell(r, 10, f'=IF(OR(F{r}="",I{r}=""),"",IF(I{r}+F{r}=0,"Push",IF((I{r}+F{r})>0,C{r},D{r})))')
        ws.cell(r, 11, f'=IF(J{r}="","",IF(J{r}="Push","Push",IF(J{r}=H{r},"Win","Loss")))')
        ws.cell(r, 12, f'=IF(F{r}="","",MAX(NORMSDIST(G{r}/$O$2),1-NORMSDIST(G{r}/$O$2)))').number_format = '0%'
        ws.cell(r, 13, record['notes'])
        for col in MANUAL_INPUT_COLUMNS:
            ws.cell(r, col).fill = manual_fill

    wb.save(output_path)
    return output_path, added, actuals_updated


def get_projection_measures(averaged_predictions, all_predictions, num_runs, game_details):
    logging.info(PROJECTION_HEADER)
    for game_index in sorted(averaged_predictions.keys()):
        predictions_array = np.array(all_predictions[game_index])
        average_pred = round(averaged_predictions[game_index] / num_runs, 1)
        std_dev = round(np.std(predictions_array), 2)
        median_pred = round(np.median(predictions_array), 1)  # still computed, not printed

        if game_index in game_details:
            info = game_details[game_index]
            logging.info(f"{game_index}|{info['week']}|{info['year']}|{info['home_team']}|{info['away_team']}|{average_pred}|{fair_line_as_home(average_pred)}|{std_dev}|{info['actual_score_difference']}")
        else:
            logging.info(f"Game {game_index}: Mean:{average_pred}|FairLine(Home):{fair_line_as_home(average_pred)}|StdDev:{std_dev}")


def translate_week_number(week_str, year):
    """Translate playoff week names to numeric values based on season year.
    
    2021+ seasons (18-game regular season):
        WildCard -> 19, Division -> 20, ConfChamp -> 21, SuperBowl -> 22
    2020 and earlier (17-game regular season):
        WildCard -> 18, Division -> 19, ConfChamp -> 20, SuperBowl -> 21
    """
    year_int = int(year)
    
    # Check if already numeric
    if week_str.isdigit():
        return int(week_str)
    
    # Playoff week translations
    if year_int >= 2021:
        # 18-game season (2021+)
        translations = {
            'WildCard': 19,
            'Division': 20,
            'ConfChamp': 21,
            'SuperBowl': 22
        }
    else:
        # 17-game season (2020 and earlier)
        translations = {
            'WildCard': 18,
            'Division': 19,
            'ConfChamp': 20,
            'SuperBowl': 21
        }
    
    return translations.get(week_str, None)


def get_last_regular_season_week(year):
    """Get the last regular season week for a given year.
    
    2021+ seasons: 18 weeks
    2020 and earlier: 17 weeks
    """
    year_int = int(year)
    return 18 if year_int >= 2021 else 17


def get_current_season_year(today=None):
    """NFL season year for today's date; January and February still belong to the season that just ended."""
    today = today or datetime.now()
    return today.year if today.month >= 3 else today.year - 1


def find_next_week_to_predict(games_past, games_future, year):
    """Earliest week of `year` that still has unplayed games.

    If the season is complete, falls back to the last week that was played.
    """
    def weeks_for(games):
        if games.empty:
            return []
        year_games = games[games['year'].astype(str) == str(year)]
        weeks = [translate_week_number(w, year) for w in year_games['week_number']]
        return [w for w in weeks if w is not None]

    upcoming = weeks_for(games_future)
    if upcoming:
        return min(upcoming)
    played = weeks_for(games_past)
    if played:
        logging.warning(f"No unplayed games left in {year}; using last played week {max(played)}")
        return max(played)
    raise ValueError(f"No games found for {year}; cannot determine the week to predict")


def run_single_iteration(iteration, cached_data, predict_year, predict_week):
    """Run a single iteration of the simulation
    
    Args:
        iteration: iteration number
        cached_data: Pre-loaded data dict with 'stats', 'all_games' DataFrames
        predict_year: Year to predict (e.g., 2025)
        predict_week: Week to predict (e.g., 21 for ConfChamp)
    """
    
    # Use pre-loaded cached data
    multi_year_combined_stats = cached_data['stats'].copy()
    all_games = cached_data['all_games'].copy()
    
    # Translate week numbers (including playoff weeks) to integers
    all_games['week_num_int'] = all_games.apply(
        lambda row: translate_week_number(row['week_number'], row['year']), axis=1
    )
    all_games['year_int'] = all_games['year'].astype(int)
    
    # Filter out rows where week translation failed (unknown week names)
    all_games = all_games[all_games['week_num_int'].notna()].copy()
    all_games['week_num_int'] = all_games['week_num_int'].astype(int)
    
    # Get games to predict: all games in the specified week/year
    games_to_predict = all_games[
        (all_games['year_int'] == predict_year) & 
        (all_games['week_num_int'] == predict_week)
    ]
    
    # Get training games: all games BEFORE the prediction week
    # Exclude the last regular season week of each year from training data
    def is_valid_training_game(row):
        game_year = row['year_int']
        game_week = row['week_num_int']
        last_reg_week = get_last_regular_season_week(game_year)
        
        # Exclude the last regular season week
        if game_week == last_reg_week:
            return False
        
        # Include if before prediction year, or same year but earlier week
        if game_year < predict_year:
            return True
        if game_year == predict_year and game_week < predict_week:
            return True
        return False
    
    training_games = all_games[all_games.apply(is_valid_training_game, axis=1)]
    
    # Only use games with scores for training
    training_games = training_games[training_games['score_difference'] != 0]

    # Prepare the training data
    merged_data, features = preprocess_data(multi_year_combined_stats, training_games)

    # Split the data into train and test sets
    X = merged_data[features]
    y = merged_data['score_difference']

    # Bootstrap resample the data for more variation
    X_resampled, y_resampled = resample(X, y, replace=True, n_samples=len(X))
    
    # Random train/test split (no random_state for different splits each iteration)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)

    # Define the models
    models = {
        'Linear Regression': LinearRegression(),
    }

    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        # Predict and evaluate the model
        y_pred = model.predict(X_test)
        score_rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        score_mae = mean_absolute_error(y_test, y_pred)
        score_r2 = r2_score(y_test, y_pred)

        results[name] = {
            'model': model,
            'cv_scores': cv_scores,
            'mean_cv_score': np.mean(cv_scores),
            'rmse': score_rmse,
            'mae': score_mae,
            'r2': score_r2
        }

    best_model_name = min(results, key=lambda k: results[k]['mae'])
    best_model = results[best_model_name]['model']

    iteration_predictions = {}
    game_info = {}  # Store game details for logging later
    
    for index, game in games_to_predict.iterrows():
        home_team = game['home_team']
        away_team = game['away_team']
        input_week = game['week_number']
        input_year = game['year']
        actual_score_difference = game['score_difference']  # May be 0 if game hasn't been played
        actual_score_difference = game['score_difference']
        
        try:
            # Get base prediction
            base_prediction = predict_game(home_team, away_team, input_year, multi_year_combined_stats, best_model)
            
            # OPTION 2: Add random noise to predictions (normal distribution with std dev of 1.0)
            noise = np.random.normal(0, 1.0)
            projected_score_difference = round(base_prediction + noise, 1)
            
            iteration_predictions[index] = projected_score_difference
            # Store game info for final logging
            game_info[index] = {
                'home_team': home_team,
                'away_team': away_team,
                'week': input_week,
                'year': input_year,
                'actual_score_difference': actual_score_difference
            }
        except ValueError as e:
            logging.error(f"Could not predict score difference for {home_team} vs {away_team}: {e}")
    
    return iteration_predictions, game_info


def main(num_runs, predict_year=None, predict_week=None, use_web=True, lookback_years=LOOKBACK_YEARS):
    """Predict one week of games.

    predict_year / predict_week default to the current season and the first week with unplayed games.
    Training data covers `lookback_years` seasons before predict_year plus predict_year itself.
    """
    if predict_year is None:
        predict_year = get_current_season_year()

    # Setup logging to file - only in main process
    log_filename = f'C:/Users/Bobby/nfl_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # Also print to console
        ],
        force=True  # Override any existing configuration
    )
    
    logging.info(f"Logging to file: {log_filename}")
    logging.info(f"Data source: {'Web URLs' if use_web else 'Local files'}")
    
    # Pre-load all data once in main process
    logging.info("Loading data from source...")
    all_years_team_stats_arr = []
    all_games_arr = []  # Combined past and future games
    predict_year_past = pd.DataFrame()
    predict_year_future = pd.DataFrame()

    for year in range(predict_year - lookback_years, predict_year + 1):
        logging.info(f"  Loading {year} data...")
        single_year_combined_stats, single_year_games_past, single_year_games_future = load_data_for_year(
            year, off_url, def_url, games_url, use_web=use_web
        )
        all_years_team_stats_arr.append(single_year_combined_stats)
        # Combine past and future games into one list
        all_games_arr.append(single_year_games_past)
        if not single_year_games_future.empty:
            all_games_arr.append(single_year_games_future)
        if year == predict_year:
            predict_year_past, predict_year_future = single_year_games_past, single_year_games_future
        
        if use_web and not _web_blocked:
            time.sleep(1)  # Rate limiting for web requests

    cached_data = {
        'stats': pd.concat(all_years_team_stats_arr, ignore_index=True),
        'all_games': pd.concat(all_games_arr, ignore_index=True)
    }
    logging.info("Data loading complete!")

    if predict_week is None:
        predict_week = find_next_week_to_predict(predict_year_past, predict_year_future, predict_year)
    logging.info(f"Predicting: Year {predict_year}, Week {predict_week}")
    
    averaged_predictions = {}
    all_predictions = {}  # Store all predictions for each game for std dev and median
    game_details = {}  # Store game details from first iteration
    
    # Determine number of worker threads (use CPU count)
    max_workers = min(32, (os.cpu_count() or 1) * 2)
    logging.info(f"Using {max_workers} worker processes")
    logging.info(f"Running {num_runs} iterations with bootstrap resampling and random noise")
    
    # Use ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all iterations - pass cached data and prediction parameters
        futures = {
            executor.submit(run_single_iteration, i, cached_data, predict_year, predict_week): i 
            for i in range(num_runs)
        }
        
        # Collect results as they complete
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            iteration = futures[future]
            completed += 1
            try:
                iteration_predictions, game_info = future.result()
                
                # Store game details from first completed iteration
                if not game_details:
                    game_details = game_info
                
                # Accumulate predictions
                for game_index, pred in iteration_predictions.items():
                    if game_index in averaged_predictions:
                        averaged_predictions[game_index] += pred
                        all_predictions[game_index].append(pred)
                    else:
                        averaged_predictions[game_index] = pred
                        all_predictions[game_index] = [pred]

                if completed % 100 == 0:
                    logging.info(f"Progress: {completed}/{num_runs} iterations completed")
                    get_projection_measures(averaged_predictions, all_predictions, completed, game_details)

            except Exception as e:
                logging.error(f"Iteration {iteration} generated an exception: {e}")

    # Calculate and log final averages with team names, std dev, and median
    logging.info("\n=== FINAL AVERAGED PREDICTIONS ===")
    logging.info(PROJECTION_HEADER)
    for game_index in sorted(averaged_predictions.keys()):
        predictions_array = np.array(all_predictions[game_index])
        average_pred = round(averaged_predictions[game_index] / num_runs, 1)
        std_dev = round(np.std(predictions_array), 2)
        median_pred = round(np.median(predictions_array), 1)  # still computed, not printed
        
        if game_index in game_details:
            info = game_details[game_index]
            logging.info(
                f"{game_index}|{info['week']}|{info['year']}|{info['home_team']}|{info['away_team']}|{average_pred}|{fair_line_as_home(average_pred)}|{std_dev}|{info['actual_score_difference']}"
            )
        else:
            logging.info(f"Game {game_index}: Mean:{average_pred}|FairLine(Home):{fair_line_as_home(average_pred)}|StdDev:{std_dev}")

    logging.info("Projections complete!")

    final_rows = []
    for game_index in sorted(averaged_predictions.keys()):
        if game_index in game_details:
            info = game_details[game_index]
            final_rows.append({
                'week': info['week'],
                'year': info['year'],
                'home_team': info['home_team'],
                'away_team': info['away_team'],
                'projected': round(averaged_predictions[game_index] / num_runs, 1),
                'actual': info['actual_score_difference'],
            })
    played_games = cached_data['all_games']
    played_games = played_games[(played_games['year'].astype(str) == str(predict_year)) & (played_games['score_difference'] != 0)]
    actual_lookup = {
        (g.week_number, g.year, g.home_team, g.away_team): g.score_difference
        for g in played_games.itertuples()
    }
    try:
        output_path, rows_added, actuals_updated = write_projections_spreadsheet(
            final_rows, predict_year, actual_lookup=actual_lookup
        )
        logging.info(f"Projections spreadsheet updated: {output_path} ({rows_added} games added, {actuals_updated} actual differences updated)")
    except Exception as e:
        logging.error(f"Could not write projections spreadsheet: {e}")


if __name__ == "__main__":
    # Configuration
    PREDICT_YEAR = None  # None = current season; set a year to override
    PREDICT_WEEK = 4  # None = first week with unplayed games; set a week number to override
    NUM_RUNS = 1#0000
    USE_WEB = True  # Set to False to use local HTML files
    
    main(
        num_runs=NUM_RUNS,
        predict_year=PREDICT_YEAR,
        predict_week=PREDICT_WEEK,
        use_web=USE_WEB
    )
