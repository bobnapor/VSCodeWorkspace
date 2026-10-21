"""One click for the weekly NFL flows: spread projections, player props and the weather-under check (wind and rain).

Each flow runs as its own Python process, exactly as if it were started by hand (python nfl_prod_projections.py and so
on): nothing is imported or shared between them, each keeps its own settings (the Configuration block at the bottom of
each script), logging and output files, and a failure in one does not stop the others.  They run one after another
rather than side by side because two of them refresh the same nflverse files in data_cache.

Double-click run_all.bat (or a shortcut to it), or run from this folder:
    python nfl_run_all.py                     every flow
    python nfl_run_all.py projections wind    only the flows named (projections, props, wind)

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
# (name, script, what it does, the Desktop file it writes)
STEPS = (
    ('projections', 'nfl_prod_projections.py', 'spread projections, picks, favorite picks and teaser legs', '{season}_nfl_projections.xlsx'),
    ('props', 'nfl_prod_props.py', 'player prop picks', '{season}_nfl_props.xlsx'),
    ('wind', 'nfl_wind_unders.py', 'weather check for totals (wind and rain unders)', '{season}_wind_unders.csv'),
)
OUTPUT_LINES = (re.compile(r'Projections spreadsheet updated: (.+?) \('), re.compile(r'Props workbook updated: (.+)$'),
                re.compile(r'Season log: (.+)$'), re.compile(r'Logging to file: (.+)$'))
PROBLEM_LINES = re.compile(r' - ERROR - |Traceback \(most recent call last\)|^\w*Error: |Could not ')


def desktop_files(names):
    """{step name: path} of the Desktop files the chosen flows write, for the open-in-Excel check; {} if unknown."""
    try:
        sys.path.insert(0, HERE)
        from nfl_sheet import get_current_season_year, get_desktop_path
        desktop, season = get_desktop_path(), get_current_season_year()
    except Exception:
        return {}
    return {name: os.path.join(desktop, pattern.format(season=season)) for name, _, _, pattern in STEPS if name in names}


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
    return '\n'.join(lines)


def main(argv):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='replace')
    known = [name for name, _, _, _ in STEPS]
    chosen = [a.lower() for a in argv] or known
    unknown = [a for a in chosen if a not in known]
    if unknown:
        print('Unknown flow(s): %s.  Choose from: %s' % (', '.join(unknown), ', '.join(known)))
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
