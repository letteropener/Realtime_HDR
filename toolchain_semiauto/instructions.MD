#Instructions for Semi-automatic toolchain



----------

##Basic steps
1. Run exposure_shots.py
2. Once the script finishes, it prompts for whether comparagram should be calculated. Enter (Y/y/yes) to process (calls system default python interpreter). If you choose not to run comparagram, immediately, mannually run comparagram_RGB_v1.py
3. After comparagrams are computed, run comparasum_RGB_iter.py to generate new comparasum

##Notes
1. These scripts rely on several txt files to operate. Modify them with caution when trying to bypass certain settings.
2. These scripts are designed to take one new set of exposure shots, do the comparagram, and generate new comparasum using previously latest comparasum and new comparagrams. If previous comparasum is not present, zero matricies are used. 