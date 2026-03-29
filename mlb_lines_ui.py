"""
mlb_lines_ui.py
---------------
Quick UI for entering MLB game matchups and odds for a given day.
Saves output to mlb_lines.csv which mlbpicks.py reads at runtime.

Usage:
    python mlb_lines_ui.py
"""

import csv
import os
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import date

# ---------------------------------------------------------------------------
# TEAM LIST
# ---------------------------------------------------------------------------
TEAMS = sorted([
    'Arizona Diamondbacks',
    'Atlanta Braves',
    'Athletics',
    'Baltimore Orioles',
    'Boston Red Sox',
    'Chicago Cubs',
    'Chicago White Sox',
    'Cincinnati Reds',
    'Cleveland Guardians',
    'Colorado Rockies',
    'Detroit Tigers',
    'Houston Astros',
    'Kansas City Royals',
    'Los Angeles Angels',
    'Los Angeles Dodgers',
    'Miami Marlins',
    'Milwaukee Brewers',
    'Minnesota Twins',
    'New York Mets',
    'New York Yankees',
    'Philadelphia Phillies',
    'Pittsburgh Pirates',
    'San Diego Padres',
    'San Francisco Giants',
    'Seattle Mariners',
    'St. Louis Cardinals',
    'Tampa Bay Rays',
    'Texas Rangers',
    'Toronto Blue Jays',
    'Washington Nationals',
])

OUTPUT_FILE = 'C:/Users/Bobby/mlb_lines.csv'
MAX_GAMES = 16

# ---------------------------------------------------------------------------
# COLOURS / STYLE
# ---------------------------------------------------------------------------
BG = '#1e1e2e'
FG = '#cdd6f4'
ENTRY_BG = '#313244'
ENTRY_FG = '#cdd6f4'
HEADER_BG = '#181825'
HEADER_FG = '#89b4fa'
BTN_SAVE = '#a6e3a1'
BTN_SAVE_FG = '#1e1e2e'
BTN_CLEAR = '#f38ba8'
BTN_CLEAR_FG = '#1e1e2e'
BTN_ADD = '#89dceb'
BTN_ADD_FG = '#1e1e2e'
SEL_BG = '#45475a'
HIGHLIGHT = '#fab387'


# ---------------------------------------------------------------------------
# GAME ROW WIDGET
# ---------------------------------------------------------------------------
class GameRow:
    """One row = one game: away team, home team, run line, ML odds, O/U."""

    def __init__(self, parent, row_num, remove_callback):
        self.frame = tk.Frame(parent, bg=BG, pady=2)
        self.frame.pack(fill='x', padx=8, pady=1)

        # Row number label
        tk.Label(
            self.frame, text=f'{row_num:>2}.',
            bg=BG, fg=HIGHLIGHT, width=3, anchor='e',
            font=('Consolas', 10, 'bold')
        ).pack(side='left', padx=(0, 4))

        # Away team
        self.away_var = tk.StringVar()
        away_cb = ttk.Combobox(
            self.frame, textvariable=self.away_var,
            values=TEAMS, state='readonly', width=22,
            font=('Consolas', 10)
        )
        away_cb.pack(side='left', padx=2)

        tk.Label(
            self.frame, text='@', bg=BG, fg=FG,
            font=('Consolas', 11, 'bold'), width=2
        ).pack(side='left')

        # Home team
        self.home_var = tk.StringVar()
        home_cb = ttk.Combobox(
            self.frame, textvariable=self.home_var,
            values=TEAMS, state='readonly', width=22,
            font=('Consolas', 10)
        )
        home_cb.pack(side='left', padx=2)

        # Separator
        tk.Label(
            self.frame, text='│', bg=BG, fg=SEL_BG,
            font=('Consolas', 11)
        ).pack(side='left', padx=4)

        # Run line (away team perspective, e.g. +1.5 or -1.5)
        tk.Label(
            self.frame, text='RL:', bg=BG, fg=HEADER_FG,
            font=('Consolas', 9)
        ).pack(side='left')
        self.rl_var = tk.StringVar(value='+1.5')
        rl_entry = tk.Entry(
            self.frame, textvariable=self.rl_var,
            bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG,
            width=6, font=('Consolas', 10),
            relief='flat', highlightthickness=1,
            highlightbackground=SEL_BG, highlightcolor=HEADER_FG
        )
        rl_entry.pack(side='left', padx=2)

        # Away ML odds
        tk.Label(
            self.frame, text='Away ML:', bg=BG, fg=HEADER_FG,
            font=('Consolas', 9)
        ).pack(side='left', padx=(6, 0))
        self.away_ml_var = tk.StringVar(value='')
        tk.Entry(
            self.frame, textvariable=self.away_ml_var,
            bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG,
            width=6, font=('Consolas', 10),
            relief='flat', highlightthickness=1,
            highlightbackground=SEL_BG, highlightcolor=HEADER_FG
        ).pack(side='left', padx=2)

        # Home ML odds
        tk.Label(
            self.frame, text='Home ML:', bg=BG, fg=HEADER_FG,
            font=('Consolas', 9)
        ).pack(side='left', padx=(6, 0))
        self.home_ml_var = tk.StringVar(value='')
        tk.Entry(
            self.frame, textvariable=self.home_ml_var,
            bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG,
            width=6, font=('Consolas', 10),
            relief='flat', highlightthickness=1,
            highlightbackground=SEL_BG, highlightcolor=HEADER_FG
        ).pack(side='left', padx=2)

        # O/U line
        tk.Label(
            self.frame, text='O/U:', bg=BG, fg=HEADER_FG,
            font=('Consolas', 9)
        ).pack(side='left', padx=(6, 0))
        self.ou_var = tk.StringVar(value='')
        tk.Entry(
            self.frame, textvariable=self.ou_var,
            bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG,
            width=6, font=('Consolas', 10),
            relief='flat', highlightthickness=1,
            highlightbackground=SEL_BG, highlightcolor=HEADER_FG
        ).pack(side='left', padx=2)

        # Away starting pitcher (last name)
        tk.Label(
            self.frame, text='Away SP:', bg=BG, fg=HEADER_FG,
            font=('Consolas', 9)
        ).pack(side='left', padx=(6, 0))
        self.away_sp_var = tk.StringVar(value='')
        tk.Entry(
            self.frame, textvariable=self.away_sp_var,
            bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG,
            width=12, font=('Consolas', 10),
            relief='flat', highlightthickness=1,
            highlightbackground=SEL_BG, highlightcolor=HEADER_FG
        ).pack(side='left', padx=2)

        # Home starting pitcher (last name)
        tk.Label(
            self.frame, text='Home SP:', bg=BG, fg=HEADER_FG,
            font=('Consolas', 9)
        ).pack(side='left', padx=(6, 0))
        self.home_sp_var = tk.StringVar(value='')
        tk.Entry(
            self.frame, textvariable=self.home_sp_var,
            bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG,
            width=12, font=('Consolas', 10),
            relief='flat', highlightthickness=1,
            highlightbackground=SEL_BG, highlightcolor=HEADER_FG
        ).pack(side='left', padx=2)

        # Remove button
        tk.Button(
            self.frame, text='✕', command=lambda: remove_callback(self),
            bg=BG, fg='#585b70', activebackground=BG,
            activeforeground=BTN_CLEAR, relief='flat',
            font=('Consolas', 10, 'bold'), cursor='hand2', bd=0
        ).pack(side='left', padx=(6, 0))

    def get_data(self):
        """Return dict of field values, or None if teams not selected."""
        away = self.away_var.get().strip()
        home = self.home_var.get().strip()
        if not away or not home:
            return None
        return {
            'away_team': away,
            'home_team': home,
            'run_line': self.rl_var.get().strip(),
            'away_ml': self.away_ml_var.get().strip(),
            'home_ml': self.home_ml_var.get().strip(),
            'ou_line': self.ou_var.get().strip(),
            'away_sp': self.away_sp_var.get().strip(),
            'home_sp': self.home_sp_var.get().strip(),
        }

    def clear(self):
        self.away_var.set('')
        self.home_var.set('')
        self.rl_var.set('+1.5')
        self.away_ml_var.set('')
        self.home_ml_var.set('')
        self.ou_var.set('')
        self.away_sp_var.set('')
        self.home_sp_var.set('')

    def destroy(self):
        self.frame.destroy()


# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------
class MLBLinesApp:

    def __init__(self, root):
        self.root = root
        self.root.title('MLB Lines Entry')
        self.root.configure(bg=BG)
        self.root.resizable(True, True)
        self.game_rows = []

        self._build_ui()
        self._add_game_row()  # start with one empty row

    # ------------------------------------------------------------------
    def _build_ui(self):
        # ---- Title bar ----
        title_frame = tk.Frame(self.root, bg=HEADER_BG, pady=8)
        title_frame.pack(fill='x')

        tk.Label(
            title_frame, text='⚾  MLB Daily Lines Entry',
            bg=HEADER_BG, fg=HEADER_FG,
            font=('Consolas', 14, 'bold')
        ).pack(side='left', padx=14)

        # Date picker on the right of title bar
        tk.Label(
            title_frame, text='Date:',
            bg=HEADER_BG, fg=FG,
            font=('Consolas', 11)
        ).pack(side='right', padx=(0, 4))

        self.date_var = tk.StringVar(value=date.today().strftime('%Y-%m-%d'))
        date_entry = tk.Entry(
            title_frame, textvariable=self.date_var,
            bg=ENTRY_BG, fg=HIGHLIGHT, insertbackground=FG,
            width=12, font=('Consolas', 11, 'bold'),
            relief='flat', highlightthickness=1,
            highlightbackground=SEL_BG, highlightcolor=HEADER_FG
        )
        date_entry.pack(side='right', padx=(0, 12))

        # ---- Column headers ----
        hdr = tk.Frame(self.root, bg=HEADER_BG, pady=4)
        hdr.pack(fill='x', padx=8, pady=(4, 0))

        headers = [
            ('#',        3,  'e'),
            ('Away Team',  24, 'w'),
            ('',          2,  'c'),   # @
            ('Home Team',  24, 'w'),
            ('',          4,  'c'),   # separator
            ('Run Line',  7, 'c'),
            ('Away ML',   7, 'c'),
            ('Home ML',   7, 'c'),
            ('O/U',       7, 'c'),
            ('Away SP',  13, 'w'),
            ('Home SP',  13, 'w'),
        ]
        for text, width, anchor in headers:
            tk.Label(
                hdr, text=text, bg=HEADER_BG, fg=HEADER_FG,
                width=width, anchor=anchor,
                font=('Consolas', 9, 'bold')
            ).pack(side='left', padx=2)

        # ---- Scrollable game rows area ----
        container = tk.Frame(self.root, bg=BG)
        container.pack(fill='both', expand=True, padx=4, pady=4)

        canvas = tk.Canvas(container, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            container, orient='vertical', command=canvas.yview
        )
        self.rows_frame = tk.Frame(canvas, bg=BG)

        self.rows_frame.bind(
            '<Configure>',
            lambda e: canvas.configure(
                scrollregion=canvas.bbox('all')
            )
        )

        canvas.create_window((0, 0), window=self.rows_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Mouse-wheel scroll
        canvas.bind_all(
            '<MouseWheel>',
            lambda e: canvas.yview_scroll(-1 * (e.delta // 120), 'units')
        )

        # ---- Bottom button bar ----
        btn_frame = tk.Frame(self.root, bg=HEADER_BG, pady=8)
        btn_frame.pack(fill='x', side='bottom')

        tk.Button(
            btn_frame, text='＋ Add Game',
            command=self._add_game_row,
            bg=BTN_ADD, fg=BTN_ADD_FG,
            font=('Consolas', 10, 'bold'),
            relief='flat', padx=12, pady=4, cursor='hand2'
        ).pack(side='left', padx=10)

        tk.Button(
            btn_frame, text='💾 Save CSV',
            command=self._save_csv,
            bg=BTN_SAVE, fg=BTN_SAVE_FG,
            font=('Consolas', 10, 'bold'),
            relief='flat', padx=12, pady=4, cursor='hand2'
        ).pack(side='left', padx=4)

        tk.Button(
            btn_frame, text='🗑 Clear All',
            command=self._clear_all,
            bg=BTN_CLEAR, fg=BTN_CLEAR_FG,
            font=('Consolas', 10, 'bold'),
            relief='flat', padx=12, pady=4, cursor='hand2'
        ).pack(side='left', padx=4)

        # Output path label
        self.status_var = tk.StringVar(value=f'Output → {OUTPUT_FILE}')
        tk.Label(
            btn_frame, textvariable=self.status_var,
            bg=HEADER_BG, fg='#6c7086',
            font=('Consolas', 9)
        ).pack(side='right', padx=12)

    # ------------------------------------------------------------------
    def _add_game_row(self):
        if len(self.game_rows) >= MAX_GAMES:
            messagebox.showinfo(
                'Max games', f'Maximum {MAX_GAMES} games per day.'
            )
            return
        row_num = len(self.game_rows) + 1
        row = GameRow(self.rows_frame, row_num, self._remove_row)
        self.game_rows.append(row)

    def _remove_row(self, row):
        self.game_rows.remove(row)
        row.destroy()
        # Renumber remaining rows
        for i, r in enumerate(self.game_rows):
            for widget in r.frame.winfo_children():
                if isinstance(widget, tk.Label) and widget.cget('fg') == HIGHLIGHT:
                    widget.config(text=f'{i+1:>2}.')
                    break

    def _clear_all(self):
        if not self.game_rows:
            return
        if messagebox.askyesno('Clear all', 'Clear all game rows?'):
            for row in self.game_rows:
                row.destroy()
            self.game_rows.clear()
            self._add_game_row()

    # ------------------------------------------------------------------
    def _validate_row(self, data, row_num):
        """Return list of error strings for a row, empty if valid."""
        errors = []
        away, home = data['away_team'], data['home_team']

        if away == home:
            errors.append(f'Row {row_num}: away and home team are the same.')

        for field, label in [
            ('run_line', 'Run Line'),
            ('away_ml', 'Away ML'),
            ('home_ml', 'Home ML'),
            ('ou_line', 'O/U'),
        ]:
            val = data[field]
            if val == '':
                continue  # optional fields allowed to be blank
            try:
                float(val)
            except ValueError:
                errors.append(
                    f'Row {row_num}: {label} "{val}" is not a number.'
                )

        return errors

    # ------------------------------------------------------------------
    def _save_csv(self):
        game_date = self.date_var.get().strip()
        try:
            from datetime import datetime
            datetime.strptime(game_date, '%Y-%m-%d')
        except ValueError:
            messagebox.showerror(
                'Bad date',
                'Date must be YYYY-MM-DD (e.g. 2025-08-20)'
            )
            return

        rows = []
        all_errors = []
        for i, row in enumerate(self.game_rows):
            data = row.get_data()
            if data is None:
                continue  # skip empty rows silently
            errs = self._validate_row(data, i + 1)
            if errs:
                all_errors.extend(errs)
            else:
                rows.append(data)

        if all_errors:
            messagebox.showerror(
                'Validation errors',
                '\n'.join(all_errors)
            )
            return

        if not rows:
            messagebox.showwarning(
                'No games',
                'No complete game rows to save. '
                'Select at least one away and home team.'
            )
            return

        # Write CSV
        fieldnames = [
            'date', 'away_team', 'home_team',
            'run_line', 'away_ml', 'home_ml', 'ou_line',
            'away_sp', 'home_sp',
        ]
        try:
            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
            with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in rows:
                    writer.writerow({
                        'date': game_date,
                        'away_team': r['away_team'],
                        'home_team': r['home_team'],
                        'run_line': r['run_line'],
                        'away_ml': r['away_ml'],
                        'home_ml': r['home_ml'],
                        'ou_line': r['ou_line'],
                        'away_sp': r.get('away_sp', ''),
                        'home_sp': r.get('home_sp', ''),
                    })
        except OSError as e:
            messagebox.showerror('Save error', str(e))
            return

        self.status_var.set(
            f'✓ Saved {len(rows)} game(s) → {OUTPUT_FILE}'
        )
        messagebox.showinfo(
            'Saved',
            f'{len(rows)} game(s) saved to:\n{OUTPUT_FILE}\n\n'
            f'Run:\n'
            f'  python mlbpicks.py --date {game_date} '
            f'--lines {OUTPUT_FILE}\n\n'
            f'Tip: Away SP / Home SP fields accept last names\n'
            f'(e.g. "Cole", "Fried") for per-game pitcher features.'
        )


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
def main():
    root = tk.Tk()
    root.geometry('1060x520')
    root.minsize(900, 300)

    # Style comboboxes
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure(
        'TCombobox',
        fieldbackground=ENTRY_BG,
        background=ENTRY_BG,
        foreground=ENTRY_FG,
        selectbackground=SEL_BG,
        selectforeground=FG,
        arrowcolor=HEADER_FG,
    )
    style.map('TCombobox', fieldbackground=[('readonly', ENTRY_BG)])
    style.configure(
        'Vertical.TScrollbar',
        background=ENTRY_BG,
        troughcolor=BG,
        arrowcolor=HEADER_FG,
    )

    MLBLinesApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
