# D^2p-fed: Differentially Private Federated Learning with Efficient Communication

This repository contains the evaluation code for the corresponding submission to ICLR'21.

### To reproduce the evaluation results

First, set up the running environment with the setup script.

```bash
./setup.sh
```

Enter the python3.7 virtual environment.
```bash
source ./venv/bin/activate
```

Enter the source repo and run the command to generate the data.
```bash
python data.py --size 10000000
```

Reproduce the evaluation results by filling in the corresponding parameters:
```bash
python simulate.py --device XX --quanlevel XX --nbit XX --dp XX --sigma2 XX
```
