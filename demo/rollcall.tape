# VHS tape — renders demo/rollcall.gif
# Usage:
#   export COUNCIL_OPENWEBUI_TOKEN='your-real-token'   # never stored in this file
#   vhs demo/rollcall.tape                              # run from repo root
# The token is read from the environment via config.yaml's `api_key_env`,
# so it never appears in the tape or the rendered frames.

Output demo/rollcall.gif

Set Shell      "bash"
Set FontSize   16
Set Width      1200
Set Height     600
Set Padding    20
Set Theme      "Catppuccin Mocha"
Set TypingSpeed 35ms

# --- Setup, hidden from the GIF ---
# Drop into a clean shell with NO rc files so Powerlevel10k (or any prompt
# theme) can't flood the frames with prompt-escape garbage, then set a tiny
# prompt and activate the venv.
Hide
Type "bash --norc --noprofile" Enter
Type `export PS1="$ "` Enter
Type "source .venv/bin/activate" Enter
Type "clear" Enter
Show

# --- Visible: just the command and its result ---
Type "council rollcall" Enter

# Let the parallel pings finish and the table settle before the GIF ends.
Sleep 5s
