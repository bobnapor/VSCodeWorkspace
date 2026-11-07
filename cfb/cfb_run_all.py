"""The daily college football run, one click: the spread picks, then the day's shareable graphic of new picks, with the
same picks as text, in CFB\\Cards on the Desktop, and the card record.

Run it once a day, around 10 AM Eastern, with no options (double-click 'CFB daily run' on the Desktop, which runs
run_all.bat).  Everything a run reads and writes is in the CFB folder on the Desktop (cfb_common.cfb_folder), which
OneDrive syncs, so any computer set up with setup.bat can run it.  As in the NFL run (nfl_run_all.py), each step runs as
its own Python process, exactly as if started by hand, and a failure in one does not stop the other; the card step only
reads what the picks step wrote.

Options, not needed day to day:
    python cfb_run_all.py picks       only the picks (then the card)
    python cfb_run_all.py card        only the card (from the workbook the picks step last wrote)
    python cfb_run_all.py nocard      the picks, no card

Everything the steps print is shown as it happens and saved to CFB\\logs\\cfb_run_all_<timestamp>.txt; a summary at the
end lists each step's result and time, any errors worth a look, and the files it updated.  Before starting, it checks
that none of the workbooks the steps write is open in Excel.
"""

import os
import re
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
# (name, script, what it does, the workbook in the CFB folder it writes that Excel could hold open)
STEPS = (
    ('picks', 'cfb_prod_picks.py', 'spread picks (the side the model favors by 2+ points)', '{season}_cfb_picks.xlsx'),
    ('card', 'cfb_share_card.py', "the day's shareable graphic and text list of new picks, and the card record",
     '{season}_cfb_card_record.xlsx'),
)
CARD, NO_CARD = 'card', 'nocard'
OUTPUT_LINES = (re.compile(r'Picks workbook updated: (.+)$'), re.compile(r'Logging to file: (.+)$'),
                re.compile(r'Share card saved: (.+)$'), re.compile(r'Pick list saved: (.+)$'),
                re.compile(r'^((?:No new picks|No picks yet|Held back) .+)$'),
                re.compile(r'(Season record of the picks: .+)$'),
                re.compile(r'^(Card record \(picks you bet\): .+)$'), re.compile(r'Card record updated: (.+)$'))
PROBLEM_LINES = re.compile(r' - ERROR - |Traceback \(most recent call last\)|^\w*Error: |Could not ')
NEXT_RUN = 'Next run: tomorrow, around 10 AM Eastern (see cfb\\README.md).'


def desktop_files(names):
    """{step name: path} of the workbooks in the CFB folder the chosen steps write; {} if unknown."""
    try:
        sys.path.insert(0, HERE)
        from cfb_common import cfb_folder, current_season
        folder, season = cfb_folder(), current_season()
    except Exception:
        return {}
    return {name: os.path.join(folder, pattern.format(season=season)) for name, _, _, pattern in STEPS if name in names}


def run_log_path():
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    try:
        sys.path.insert(0, HERE)
        from cfb_common import cfb_folder
        folder = cfb_folder('logs')
    except Exception:
        folder = os.path.expanduser('~')
    return os.path.join(folder, f'cfb_run_all_{stamp}.txt')


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


def run_step(name, script, log):
    """Run one step in its own process, echoing and logging its output; returns what the summary needs."""
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    started = time.time()
    outputs, problems, warnings = [], [], 0
    header = '=' * 20 + f' {name}: python {script} (started {datetime.now():%H:%M:%S}) ' + '=' * 20
    echo('\n' + header)
    log.write('\n' + header + '\n')
    process = subprocess.Popen([sys.executable, '-W', 'ignore', script], cwd=HERE, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
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
    return {'name': name, 'code': code, 'seconds': time.time() - started, 'outputs': outputs, 'problems': problems,
            'warnings': warnings}


def summarize(results, log_path, started):
    lines = ['', '=' * 30 + f' Summary ({duration(time.time() - started)} in all) ' + '=' * 30]
    for r in results:
        status = 'OK' if r['code'] == 0 else f"FAILED (exit code {r['code']})"
        extra = f", {r['warnings']} warnings in its log" if r['warnings'] else ''
        lines.append('%-8s %-22s %s%s' % (r['name'], status, duration(r['seconds']), extra))
        lines += ['%8s %s' % ('', output) for output in r['outputs']]
        lines += ['%8s ! %s' % ('', problem[:160]) for problem in r['problems'][:5]]
        if len(r['problems']) > 5:
            lines.append('%8s ! ... %d more, see the full output' % ('', len(r['problems']) - 5))
    lines.append(f'Full output: {log_path}')
    lines.append(NEXT_RUN)
    return '\n'.join(lines)


def chosen_steps(argv):
    """The step names to run, in STEPS order, or an error message: the picks then the card by default, the card after
    the picks (unless nocard), and the card alone when it is the only one named."""
    known = [name for name, _, _, _ in STEPS]
    args = [a.lower() for a in argv]
    named = [a for a in args if a != NO_CARD]
    unknown = [a for a in named if a not in known]
    if unknown:
        return f"Unknown step(s): {', '.join(unknown)}.  Choose from: {', '.join(known)} (and {NO_CARD})"
    chosen = set(named or known) | {CARD}
    if NO_CARD in args:
        chosen.discard(CARD)
    return [name for name in known if name in chosen] or 'Nothing to run.'


def main(argv):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='replace')
    chosen = chosen_steps(argv)
    if isinstance(chosen, str):
        print(chosen)
        return 2
    steps = [s for s in STEPS if s[0] in chosen]
    started = time.time()
    log_path = run_log_path()
    print('Running ' + ', '.join(f'{n} ({what})' for n, _, what, _ in steps) + ', one after another.')
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
