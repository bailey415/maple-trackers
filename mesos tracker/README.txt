MESOS TRACKER
=============

Tracks the mesos YOUR character picks up by reading the meso total at the
bottom of your inventory window. Keep the inventory open (any tab) and on
screen while you play.

HOW TO USE
  1. Double-click run.bat (first run installs the libraries it needs).
  2. Your browser opens http://127.0.0.1:8766 - that's the dashboard.
     (The exp tracker uses 8765, so both can run at the same time.)
  3. Leave the black window open while you play. Close it to stop.

FIRST FEW MINUTES - TEACHING DIGITS
  The tracker has only seen the digits 3, 7 and 1 on your screen so far.
  The others start as close guesses. While a guess is in use, the
  dashboard shows a picture of your meso box and asks
  "Does your inventory say 37,240?"
    - Click Yes if it's right. Every digit in that number is learned.
    - If it's wrong, type the real number and click Fix.
  Once all ten digits are learned (saved in meso_font.json) it never asks
  again.

WHAT IT SHOWS
  Mesos/hour    - mesos picked up per hour over the last 5/10/15/30 minutes
  Today's mesos - picked up since midnight, plus spent and net
  Spent         - mesos spent today (shops, etc.)
  Your current total, a sparkline, and the status dot
  ("Inventory on screen" / "Inventory not visible") are always shown.

NOTES
  - Spending never lowers your mesos/hour; it's shown separately.
  - Time when the inventory is closed or covered doesn't count toward
    mesos/hour.
  - Every reading is saved in the data folder (one CSV per day), so closing
    and reopening keeps today's totals.
