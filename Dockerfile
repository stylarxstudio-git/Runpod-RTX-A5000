FROM runpod/base:0.6.1-cuda12.1.0-py3.10

COPY requirements.txt /requirements.txt
COPY rp_handler.py /rp_handler.py

RUN pip install --upgrade pip && pip install -r /requirements.txt

CMD [ "python", "-u", "/rp_handler.py" ]
