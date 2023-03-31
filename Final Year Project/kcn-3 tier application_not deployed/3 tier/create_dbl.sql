CREATE DATABASE eyeHospital;
USE eyeHospital;

# Create the user which the web app will use to access the database
DROP USER IF EXISTS 'hospitalapp'@'localhost';
CREATE USER 'hospitalapp'@'localhost' IDENTIFIED WITH mysql_native_password BY 'h0sp1t4l';
GRANT ALL privileges ON eyeHospital.* TO 'hospitalapp'@'localhost';

# Remove the tables if they already exist
DROP TABLE IF EXISTS patients;
DROP TABLE IF EXISTS services;
DROP TABLE IF EXISTS eyeconditions;


# Create the patients to store patient details
CREATE TABLE Patients (
  patient_id INT NOT NULL AUTO_INCREMENT,
  firstname VARCHAR(255) NOT NULL,
  surname VARCHAR(255) NOT NULL,
  dob DATE,
  sex VARCHAR(255) NOT NULL,
  race VARCHAR(255) NOT NULL,
  ethnicity VARCHAR(255),
  PRIMARY KEY(patient_id)
);

# Create the Services table to store the list of available Services
CREATE TABLE Services (
   Service_id INT NOT NULL AUTO_INCREMENT,
   Service VARCHAR(255),
   PRIMARY KEY(Service_id)
);

# Create the EyeConditionstable to store the list of available Eye Conditions
CREATE TABLE EyeConditions(
   EyeCondition_id INT NOT NULL UNIQUE AUTO_INCREMENT,
   EyeCondition_name VARCHAR(255),
   PRIMARY KEY(EyeCondition_id)
);

CREATE TABLE MedicalRecord (
mr_id INT NOT NULL UNIQUE AUTO_INCREMENT,
Hospital_Number INT(6) NOT NULL UNIQUE,
patient_id INT NOT NULL,
Appointment_History TEXT,
Medical_Image BLOB,
PRIMARY KEY (mr_id),
FOREIGN KEY (patient_id) REFERENCES Patients(patient_id)
);

CREATE TABLE Doctors (
employee_id INT NOT NULL UNIQUE AUTO_INCREMENT,
jobTitle TEXT NOT NULL,
GMC_id INT (8) NOT NULL UNIQUE,
Specialty TEXT NOT NULL,
patient_id INT,
PRIMARY KEY (employee_ID),
FOREIGN KEY (patient_id) REFERENCES Patients(patient_id)
);

/* CREATE TABLE Admissions (
   admission_id INT NOT NULL AUTO_INCREMENT,
   admission_date DATETIME,
   patient_id INT,
   EyeCondition_id INT,
   PRIMARY KEY (admission_id),
   FOREIGN KEY (patient_id) REFERENCES Patients(patient_id),
   FOREIGN KEY (EyeCondition_id) REFERENCES EyeConditions(EyeCondition_id)
);

CREATE TABLE Diagnosis (
   patient_id INT,
   diagnosis_time DATETIME,
   EyeCondition_name VARCHAR(50),
   FOREIGN KEY (EyeCondition_name) REFERENCES EyeConditions(EyeCondition_name),
   FOREIGN KEY (patient_id) REFERENCES patients(patient_id)


); */