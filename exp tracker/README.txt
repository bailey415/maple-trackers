EXP TRACKER
===========

Tracks YOUR character's EXP by reading the numbers on your own status bar
(the "EXP 168,840 [36.04%]" text next to HP/MP). Nothing else on screen is
read, so other players never count.

HOW TO USE
  1. Double-click run.bat (first run installs the libraries it needs).
  2. Your browser opens http://127.0.0.1:8765 - that's the dashboard.
  3. Click the LV badge once and type your current level.
     (Click the name to rename it.) After that the level goes up by itself
     whenever the EXP % rolls over.
  4. Leave the black window open while you play. Close it to stop.

WHAT IT SHOWS
  EXP/hour   - EXP per hour and % of a level per hour, over the last
               5 / 10 / 15 / 30 minutes of play
  Today's EXP- everything gained since midnight (deaths are subtracted)
  Level      - level, EXP left to the next level, time until you level,
               and today's level-ups
  The bar, the "today · next level in" line and the status dot
  ("On screen" / "Off screen") are always visible. The page updates every 30s.

NOTES
  - The game has to be visible on screen (GeForce NOW can be fullscreen or
    windowed). If another window covers the status bar, it shows Off screen
    and that time doesn't count toward EXP/hour.
  - Every reading is saved in the data folder (one CSV per day), so closing
    and reopening the tracker keeps today's totals.
  - Settings are in tracker_settings.json (name, level, port).
