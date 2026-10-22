"""The daily NFL run, one click: spread projections, player props and the weather-under check (wind and rain), then
the day's shareable graphic of new picks, with the same picks as text, in the 'NFL picks' folder on the Desktop.

Run it once a day, around 10 AM Eastern, with no options (double-click the 'NFL daily run' shortcut or run_all.bat):
every flow picks only what is worth betting at that moment (overs early in the week, unders within 24 hours of
kickoff, weather from a near-kickoff forecast), and the card shows only picks no earlier day's card showed.  The
schedule, day by day, is at the top of README.md.

Each flow runs as its own Python process, exactly as if it were started by hand (python nfl_prod_projections.py and so
on): nothing is imported or shared between them, each keeps its own settings (the Configuration block at the bottom of
each script), logging and output files, and a failure in one does not stop the others.  They run one after another
rather than side by side because two of them refresh the same nflverse files in data_cache.  The last step,
nfl_share_card.py, only reads what the flows wrote (it changes none of their logic or files).

Options, not needed day to day:
    python nfl_run_all.py projections wind    only the flows named (projections, props, wind), then the card
    python nfl_run_all.py card                only the card (from the files the flows last wrote)
    python nfl_run_all.py nocard              every flow, no card (nocard works with flow names too)

Everything the flows print is shown as it happens and saved to ~/nfl_run_all_<timestamp>.txt.  A summary at the end
lists each flow's result and time, any errors worth a look, and the files it updated.  Before starting, it checks that
none of the files the flows write is open in Excel (an open workbook cannot be saved).
"""

import os
import re
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
# (name, script, what it does, the Desktop file it writes that Excel could hold open); the card comes last, as it reads
# what the flows before it wrote
STEPS = (
    ('projections', 'nfl_prod_projections.py', 'spread projections, picks, favorite picks and teaser legs', '{season}_nfl_projections.xlsx'),
    ('props', 'nfl_prod_props.py', 'player prop picks', '{season}_nfl_props.xlsx'),
    ('wind', 'nfl_wind_unders.py', 'weather check for totals (wind and rain unders)', '{season}_wind_unders.csv'),
    ('card', 'nfl_share_card.py', "the day's shareable graphic and text list of new picks", None),
)
CARD, NO_CARD = 'card', 'nocard'
OUTPUT_LINES = (re.compile(r'Projections spreadsheet updated: (.+?) \('), re.compile(r'Props workbook updated: (.+)$'),
                re.compile(r'Season log: (.+)$'), re.compile(r'Logging to file: (.+)$'), re.compile(r'Share card saved: (.+)$'),
                re.compile(r'Pick list saved: (.+)$'), re.compile(r'^((?:No new picks|No picks yet|Held back) .+)$'))
PROBLEM_LINES = re.compile(r' - ERROR - |Traceback \(most recent call last\)|^\w*Error: |Could not ')
NEXT_RUN = 'Next run: tomorrow, around 10 AM Eastern (the daily schedule is at the top of nfl\\README.md).'


def desktop_files(names):
    """{step name: path} of the Desktop files the chosen flows write, for the open-in-Excel check; {} if unknown."""
    try:
        sys.path.insert(0, HERE)
        from nfl_sheet import get_current_season_year, get_desktop_path
        desktop, season = get_desktop_path(), get_current_season_year()
    except Exception:
        return {}
    return {name: os.path.join(desktop, pattern.format(season=season)) for name, _, _, pattern in STEPS
            if name in names and pattern}


def locked(paths):
    """The paths that exist and cannot be opened for writing (open in Excel)."""
    found = []
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path, 'a'):
                    pass
            except PermissionError:
                found.append(path)
    return found


def check_files_closed(names):
    paths = list(desktop_files(names).values())
    still = locked(paths)
    if still and sys.stdin is not None and sys.stdin.isatty():
        print('These files are open (probably in Excel) and could not be saved:\n  ' + '\n  '.join(still))
        input('Close them, then press Enter to continue (Ctrl+C to stop)... ')
        still = locked(paths)
    if still:
        print('Still open, so these will not be saved this run:\n  ' + '\n  '.join(still) + '\n')


def duration(seconds):
    return '%dm %02ds' % divmod(int(round(seconds)), 60)


def echo(text):
    """Print a line even when the console's encoding cannot show every character in it."""
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or 'ascii'
        print(text.encode(encoding, errors='replace').decode(encoding), flush=True)


def run_step(name, script, log, cwd=HERE):
    """Run one flow in its own process, echoing and logging its output; returns what the summary needs."""
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    started = time.time()
    outputs, problems, warnings = [], [], 0
    header = '=' * 20 + ' %s: python %s (started %s) ' % (name, script, datetime.now().strftime('%H:%M:%S')) + '=' * 20
    echo('\n' + header)
    log.write('\n' + header + '\n')
    process = subprocess.Popen([sys.executable, script], cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, env=env)
    try:
        for raw in process.stdout:
            line = raw.decode('utf-8', errors='replace').rstrip('\r\n')
            echo(line)
            log.write(line + '\n')
            for pattern in OUTPUT_LINES:
                match = pattern.search(line)
                if match:
                    outputs.append(match.group(1).strip())
            if ' - WARNING - ' in line:
                warnings += 1
            elif PROBLEM_LINES.search(line):
                problems.append(line.strip())
        code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise
    log.flush()
    return {'name': name, 'code': code, 'seconds': time.time() - started, 'outputs': outputs, 'problems': problems, 'warnings': warnings}


def summarize(results, log_path, started):
    lines = ['', '=' * 30 + ' Summary (%s in all) ' % duration(time.time() - started) + '=' * 30]
    for r in results:
        status = 'OK' if r['code'] == 0 else 'FAILED (exit code %d)' % r['code']
        extra = ', %d warnings in its log' % r['warnings'] if r['warnings'] else ''
        lines.append('%-12s %-22s %s%s' % (r['name'], status, duration(r['seconds']), extra))
        for output in r['outputs']:
            lines.append('%12s %s' % ('', output))
        for problem in r['problems'][:5]:
            lines.append('%12s ! %s' % ('', problem[:160]))
        if len(r['problems']) > 5:
            lines.append('%12s ! ... %d more, see the full output' % ('', len(r['problems']) - 5))
    lines.append('Full output: %s' % log_path)
    lines.append(NEXT_RUN)
    return '\n'.join(lines)


def chosen_steps(argv):
    """The step names to run, in STEPS order, or an error message: every flow by default, the card after any flows
    (unless nocard), and the card alone when it is the only one named."""
    known = [name for name, _, _, _ in STEPS]
    args = [a.lower() for a in argv]
    named = [a for a in args if a != NO_CARD]
    unknown = [a for a in named if a not in known]
    if unknown:
        return 'Unknown flow(s): %s.  Choose from: %s (and %s)' % (', '.join(unknown), ', '.join(known), NO_CARD)
    chosen = set(named or known) | {CARD}
    if NO_CARD in args:
        chosen.discard(CARD)
    if not chosen:
        return 'Nothing to run.'
    return [name for name in known if name in chosen]


def main(argv):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='replace')
    chosen = chosen_steps(argv)
    if isinstance(chosen, str):
        print(chosen)
        return 2
    steps = [s for s in STEPS if s[0] in chosen]
    started = time.time()
    log_path = os.path.join(os.path.expanduser('~'), 'nfl_run_all_%s.txt' % datetime.now().strftime('%Y%m%d_%H%M%S'))
    print('Running %s, one after another (each in its own process).' % ', '.join('%s (%s)' % (n, what) for n, _, what, _ in steps))
    check_files_closed(chosen)
    results = []
    with open(log_path, 'w', encoding='utf-8') as log:
        try:
            for name, script, _, _ in steps:
                results.append(run_step(name, script, log))
        except KeyboardInterrupt:
            print('\nStopped by Ctrl+C.')
        summary = summarize(results, log_path, started)
        echo(summary)
        log.write(summary + '\n')
    finished = len(results) == len(steps)
    return 0 if finished and all(r['code'] == 0 for r in results) else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
