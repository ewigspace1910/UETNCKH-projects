FROM python:3.9-slim-buster
RUN apt-get update && apt-get install \
libglib2.0-0 \
libgl1-mesa-glx -y && \
apt-get autoremove && \
rm -rf /var/lib/apt/lists/* && \
rm -rf /var/cache/apt/archives

RUN mkdir -p /app
RUN mkdir -p /tmp
WORKDIR /app

# Copy requirements and install them
COPY ./requirements.txt ./requirements.txt
RUN python3 -m pip install --no-cache-dir -r ./requirements.txt

# Download NLTK data
RUN python3 -m nltk.downloader punkt
RUN [ "python3", "-c", "import nltk; nltk.download('inference')"]
RUN [ "python3", "-c", "import nltk; nltk.download('parse')"]
RUN [ "python3", "-c", "import nltk; nltk.download('chat')"]
RUN [ "python3", "-c", "import nltk; nltk.download('chunk')"]
RUN [ "python3", "-c", "import nltk; nltk.download('stopwords')"]
RUN [ "python3", "-c", "import nltk; nltk.download('punkt')"]
RUN [ "python3", "-c", "import nltk; nltk.download('averaged_perceptron_tagger')"]

RUN cp -r /root/nltk_data /usr/share/nltk_data
RUN rm -r /root/nltk_data

# Copy the rest of the application
COPY ./db_integration ./db_integration
COPY ./Modules        ./Modules
COPY ./Routers        ./Routers
COPY ./config.py      .
COPY ./main.py        .
COPY ./.env           .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--workers", "4"]