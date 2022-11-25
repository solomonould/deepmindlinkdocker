FROM python:latest
# Or any preferred Python version.
WORKDIR /usr/app/src
ADD MV_LSTM_checkpoints ./
ADD data ./
ADD model_file.py .
ADD main.py .
RUN pip install requests torch numpy scikit-learn flask
EXPOSE 2052
COPY main.py ./
CMD python3 main.py 
# Or enter the name of your unique directory and parameter set.