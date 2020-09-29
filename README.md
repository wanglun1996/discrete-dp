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

Reproduce the evaluation results by filling in the corresponding parameters:
```bash
python simulate.py --device XX --quanlevel XX --nbit XX --dp XX --sigma2 XX
```
