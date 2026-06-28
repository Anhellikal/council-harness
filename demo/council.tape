# VHS tape — renders demo/council.gif
# Usage:  vhs demo/council.tape      (run from repo root, with Ollama up)
# If your agents need a token:  export COUNCIL_OPENWEBUI_TOKEN='...' first.

Output demo/council.gif

Set Shell      "bash"
Set FontSize   16
Set Width      1200
Set Height     720
Set Padding    20
Set Theme      "Catppuccin Mocha"
Set TypingSpeed 35ms

# --- Setup, hidden from the GIF ---
# Clean shell (no rc) so Powerlevel10k / prompt themes can't flood the frames,
# then a tiny prompt + venv.
Hide
Type "bash --norc --noprofile" Enter
Type `export PS1="$ "` Enter
Type "source .venv/bin/activate" Enter
Type "clear" Enter
Show

# --- Visible: the command and the live dashboard ---
Type `council run "write a python function that checks if a string is a palindrome" --config demo/config.demo.yaml --dashboard` Enter

# Set this to ROUGHLY one real run's wall-clock time so the GIF ends on the
# finished dashboard, not mid-round. Time a real run first, then tune.
Sleep 35s
