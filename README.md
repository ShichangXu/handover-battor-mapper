# handover-battor-mapper

This repository contains code borrowed from another project that will allow you to synchronize handover logs with battor logs.

The `handover_energy.py` is a minimal working example of how to associate handover's `os_timestamp` or `diag_timestamp` with its corresponding battor logline.

Please note that this project is in no way complete.
For example, currently the project only parses RRC, ServingCell, NeighborCell, RRCStateChange logs from the entire handover log file.
The reason most other log types were ignored was because they did not follow what we considered to be the standard format

    os_timestamp    diag_timestamp
    payload

