# How Much Gold Should an Indian Investor Hold?

Python pipeline + Streamlit dashboard for the equity-gold-bond portfolio optimization project.

## Project structure

```
gold_portfolio_project/
├── data/
│   └── Portfolio_RawData_Input.xlsx   <- raw prices only, one sheet per asset
├── config.py                          <- every constant lives here (assets, floor, regime dates)
├── data.py                            <- loading, returns, CPI-adjusted real returns, dataset assembly
├── optimizer.py                       <- min-variance, tangency, efficient frontier, rolling window
├── app.py                             <- Streamlit dashboard
├── requirements.txt
└── README.md                          <- this file
```

**Why split this way:** `config.py` holds every knob (which sheet is "gold", where regime
boundaries fall, the allocation floor) so changing an assumption never means hunting through
calculation code. `data.py` only produces clean return series — it does no optimization.
`optimizer.py` only does optimization — it doesn't know where the numbers came from. `app.py`
just wires the two together for display. This means you can test `data.py` and `optimizer.py`
completely on their own (see below), which is exactly how bugs get caught before they propagate
into a rolling-window loop where they're much harder to spot.

## Setup

```bash
cd gold_portfolio_project
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running things

**Sanity-check the data pipeline** (prints regime counts, risk-free coverage, a few rows):
```bash
python data.py
```

**Sanity-check the optimizer** (prints full-sample mean/cov, min-variance and tangency weights,
last 5 rows of the rolling window):
```bash
python optimizer.py
```

**Launch the dashboard:**
```bash
streamlit run app.py
```
This opens in your browser automatically (usually `http://localhost:8501`).

---

## Navigating this project in VS Code

### Opening the project
- `File > Open Folder...` and select the `gold_portfolio_project` folder (not a single file —
  opening the folder is what lets VS Code find `config.py` etc. as importable modules, and
  lets the Python extension detect your virtual environment).

### The three panels you'll use constantly
- **Explorer** (left sidebar, top icon, or `Ctrl+Shift+E` / `Cmd+Shift+E`): the file tree. Click
  any `.py` file to open it.
- **Editor** (center): where you read/write code. Multiple files open as tabs along the top,
  same as a browser.
- **Integrated Terminal** (`` Ctrl+` `` / `` Cmd+` ``, or `View > Terminal`): a real terminal,
  already `cd`'d into your project folder. This is where you run `python data.py`,
  `streamlit run app.py`, `pip install`, etc. — no need to switch to a separate terminal app.

### Jumping around code fast
- `Ctrl+P` / `Cmd+P` ("Quick Open"): type a filename to jump straight to it — much faster than
  clicking through the Explorer once you have more than a few files.
- `Ctrl+Shift+P` / `Cmd+Shift+P` ("Command Palette"): the "do anything" search bar — e.g. type
  "Python: Select Interpreter" to point VS Code at your venv (see below).
- `F12` (or `Ctrl+Click` on Windows/Linux, `Cmd+Click` on Mac) on a function name: jumps to
  where it's *defined*. Try it on a call to `optimizer.tangency(...)` from `app.py`.
- `Shift+F12`: the reverse — shows every place a function is *used*. Useful before renaming
  anything in `config.py`, to see everywhere it's referenced.
- `Ctrl+Shift+O` / `Cmd+Shift+O`: jump to any function/class *within the current file* — handy
  in `optimizer.py` once it has a dozen functions.

### Selecting the right Python interpreter (important!)
VS Code needs to know which Python (and which installed packages) to use. After creating the
venv above:
1. `Ctrl+Shift+P` → "Python: Select Interpreter"
2. Choose the one whose path includes `venv` (e.g. `./venv/bin/python`)

If this is set wrong, you'll see red squiggly underlines under `import pandas` etc. even though
`pip install` succeeded — it just means VS Code is looking at a different Python than the one
you installed packages into.

### Running code without leaving the editor
- Click the **▷ Run** arrow in the top-right of the editor (or `Ctrl+F5`) to run whichever
  `.py` file is currently open, without typing the command yourself.
- For `app.py` specifically, you still need `streamlit run app.py` in the terminal — Streamlit
  apps aren't run the normal way, since they're a whole server, not a one-shot script.

### Debugging (stepping through code line by line)
1. Click just left of a line number to set a **breakpoint** (a red dot appears) — e.g. inside
   `optimizer.tangency()`.
2. Press `F5` (or the Run and Debug icon in the left sidebar, then "Run and Debug"). First time,
   VS Code asks what kind of file — choose "Python File".
3. Execution pauses at your breakpoint. Use the floating controls (or `F10` step over, `F11`
   step into) to move line by line, and hover over any variable to see its current value.

This is the fastest way to answer "why is this weight coming out wrong" — much faster than
sprinkling `print()` statements everywhere, and it's exactly how you'd catch a row-offset or
sign-flip bug like the ones we found in the Excel version, before it silently propagates.

### Two small quality-of-life settings
- `Ctrl+Shift+P` → "Preferences: Open User Settings" → search "format on save" and enable it —
  keeps every file consistently indented without thinking about it.
- Install the **Python** extension (by Microsoft) and **Pylance** if not already present —
  `Ctrl+Shift+X` opens the Extensions panel, search "Python". These give you the red-squiggle
  error checking, autocomplete, and the "jump to definition" behavior above.

## Known data limitations (carried over from the source workbook)

- **91-Day T-Bill** only covers Nov-2018 to Mar-2025. Any regime or rolling window outside that
  range will have a partial or missing risk-free rate — `data.py` and `optimizer.py` surface
  this explicitly (`rf_coverage` strings, `None` returns) rather than silently guessing.
- **CPI base-year change**: MOSPI switched from 2012=100 to 2024=100 starting Jan-2026.
  `data.monthly_inflation()` handles this correctly already — see the docstring — but if you
  ever compute inflation a different way, re-check this boundary specifically.
- **Nippon Gold BeES, Oct-2016** had a decimal-shift data error in the original source
  (2784.2 instead of 27.84) — already corrected in `data/Portfolio_RawData_Input.xlsx`.
- With the default 25% floor, the tangency portfolio equals the minimum-variance portfolio in
  every regime we tested, because the real risk-free rate has been persistently negative. This
  is a genuine finding worth discussing in your write-up, not a bug — try the floor slider at
  0% in the dashboard to see the unconstrained, purely data-driven picture underneath it.
