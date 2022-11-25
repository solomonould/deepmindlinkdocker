FROM python:latest
# Or any preferred Python version.
WORKDIR /usr/app/src
ADD MV_LSTM_checkpoints /usr/app/src
ADD data /usr/app/src
ADD model_file.py /usr/app/src
ADD main.py /usr/app/src
RUN pip install requests torch numpy scikit-learn flask
EXPOSE 2052
CMD python3 /usr/app/src/main.py 
# Or enter the name of your unique directory and parameter set.