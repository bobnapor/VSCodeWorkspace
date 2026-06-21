# Fantasy Football History Web App

A Flask + SQLite starter app for storing and exploring fantasy football league history.

## What it includes

- League dashboard with all-time standings
- Season history table
- Career records leaderboards
- Analytics page with Chart.js visuals
- Head-to-head explorer
- Trade log
- CSV import page for Google Sheets exports
- Seeded demo data based on the visible structure of the provided `Skimbleshanks` spreadsheet

## Run locally

```bash
cd fantasy_football_web
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000

## CSV import formats

### Team summaries

```csv
year,owner,team,wins,losses,ties,pf,pa,rank,final_place,made_playoffs,moves
2023,Fred,"Oy Vey, JJ!",7,7,0,1667,1752,1,1,true,23
```

### Trades

```csv
year,week,from_owner,to_owner,asset_name,asset_type,notes
2023,5,Fred,Ben,Tyreek Hill,player,Needed RB depth
```

## Notes

- The app seeds demo data only when the database is empty.
- Importing CSV lets you replace the demo with real league data over time.
- If you want to start over, delete `fantasy_football.db` and rerun the app.
