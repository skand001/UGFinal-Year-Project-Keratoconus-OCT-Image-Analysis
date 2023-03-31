USE eyeHospital;

DROP VIEW IF EXISTS pw;

CREATE VIEW pw AS
    SELECT Patients_patient_id, Patients.firstname, Patients_surname, Patients_pdob, Patients_sex, Patients_race, Patients_ethnicity
    FROM Patients  
    JOIN EyeConditions ON EyeCondition_id 


/* IT WORKS */
    CREATE VIEW pw AS
    SELECT * 
    FROM Patients p
    JOIN Admissions a 
    ON p.patient_id = a.patient_id
    ORDER BY admission_date DESC

    CREATE VIEW full_view AS
    SELECT * 
    FROM Patients p
    JOIN Admissions a 
    ON p.patient_id = a.patient_id
    LEFT JOIN EyeConditions e 
    ON a.patient_id = e.patient_id
    ORDER BY admission_date ASC;

    ALTER TABLE EyeConditions DROP FOREIGN KEY patient_id;