# Hacking a Chat Bot

## Automated Testing

### Garak

Reference: https://github.com/NVIDIA/garak
Reference: https://reference.garak.ai/en/latest/extending.generator.html
Note - the garak install is BIG!! It will pull a bunch of libraries which are large and this will take a lot of time.



#### Setup Garak

You want to do this section at home first. This will bring in all of the massive pre-reqs that Garak has.
```bash
python3 -m venv ./garak-example-venv
source ./garak-example-venv/bin/activate
git clone https://github.com/NVIDIA/garak garak-repo
cd garak-repo
python -m pip install -e .
```
