curl -X POST -F "file=@D:/1/work/SnapLync/whatsapp.m4a" http://67.205.185.110:8080/upload/audio/ 

## Download vosk model from google drive
gdown https://drive.google.com/uc?id=1BqxDuiQhsirClKuiJB3zBzeFXgLL5WLx 


## Run fastapi app
uvicorn main:app --host 0.0.0.0 --port 8080


## password for DigitalOcean Droplet instance
123qwe!@#QWE


## IP address for DigitalOcean Droplet instance
67.205.185.110

## Make uvicron app as a linux service
https://blog.miguelgrinberg.com/post/running-a-flask-application-as-a-service-with-systemd

## Database

Name : SpeechDoctorDB
user : root

Tables : Category(id, category), Category_Question(category_id, question_id), UserInfo(id, phone, email), User_Category(user_id, category_id)