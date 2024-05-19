Simple RAG implementation with langchain and various models. 

The nearest-neighbour search isn't working as well as it should.

# Install

Set up a venv and install the requirements

```sh
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

download the en_core_web_sm

```sh
python3 -m spacy download en_core_web_sm
```

# Running
```sh
python3 -W ignore app.py`
```
