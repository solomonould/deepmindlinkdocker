FROM python:latest
# Or any preferred Python version.
WORKDIR /usr/app/src
COPY MV_LSTM_checkpoints /usr/app/src/MV_LSTM_checkpoints
COPY data /usr/app/src/data
COPY model_file.py /usr/app/src
COPY main.py /usr/app/src
RUN pip install requests torch numpy scikit-learn flask
EXPOSE 2052
CMD python3 /usr/app/src/main.py 
# Or enter the name of your unique directory and parameter set.