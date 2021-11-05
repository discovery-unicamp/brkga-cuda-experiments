#!/bin/bash

echo "Logging..."
ssh -t win-aspire "
    cd .\projects\BRKGA\ ;
    echo 'Sync sources...' ;
    wsl -- rsync -q -razv --exclude-from=.rsync-exclude aspire:~/projects/facul/BRKGA/ . ;
    wsl -- ./run.sh $@ ;
    "
