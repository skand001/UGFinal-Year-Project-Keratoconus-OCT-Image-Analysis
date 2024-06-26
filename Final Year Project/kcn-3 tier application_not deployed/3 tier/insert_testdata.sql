use eyeHospital;

INSERT INTO Patients (firstname, surname, dob, sex, race)
VALUES
('Vernon', 'Lopez', '1956-11-01', 'Male', 'White'),
('Bethany', 'Tanner','1989-03-11', 'Female',  'Asian'),
('Nadine', 'Burnett','1956-02-27', 'Female',  'Black African'),
('Margo', 'Ortega','1978-04-30', 'Female',  'Asian'),
('Gustavo', 'Newman','1985-04-15', 'Male',  'Mixed'),
('Amelia', 'Aguilar','1967-06-03', 'Female',  'Asian'),
('Roberta', 'Acosta','1955-11-05', 'Female',  'Mixed'),
('Willard', 'French','1990-12-06', 'Male',  'Asian'),
('Denver', 'Owen','1989-09-09', 'Female',  'Asian'),
('Edison', 'Flowers','1949-08-13', 'Male',  'Black African'),
('Elisa', 'Stuart','1983-07-23', 'Female',  'White'),
('Tracey', 'Mclean','1986-01-25', 'Female',  'Asian'),
('Briana', 'Pham','1998-03-18', 'Female',  'White'),
('Earl ', 'Reed','2000-04-01', 'Male',  'Asian'),
('Randell', 'Wood','1999-09-30', 'Female', 'White');




INSERT INTO Services (Service)
VALUES  
('Adnexal'),
('Cataract'),
('Contact Lense'),
('External Disease'),
('Corneal Disease'),
('Glaucoma'),
('Medical Retina'),
('Ocular Oncology'),
('Ocular Prosthetics'),
('Pharmacy'),
('Radiology'),
('Patient Advise and Liason (PALS)'),
('Childrens services- paediatrics'),
('General Opthalmology (LVA)'),
('Refracive Surgery'),
('Retinal Therapy Unit'),
('Spectacle Dispensing'),
('Strabismus and Neuro-Opthalmology'),
('Vitroretinal');


INSERT INTO EyeConditions (EyeCondition_name)
VALUES 
('Acanthamoeba Keratitis'),
('Achromatopsia'),
('Age-related macular degeneration'),
('Amblyopia (lazy eye)'),
('Astigmatism'),
('Birdshot chorioretinopathy'),
('Blepharatis'),
('Cataract'),
('Chalaizon'),
('Conjuctivitis'),
('Corneal abrasion'),
('Diabetic macular oedema'),
('Diabetic retinopathy'),
('Endophtalmitits'),
('Epiretinal membrane'),
('Episcleritis'),
('Flashes and floaters'),
('Fuchs dystrophy'),
('Glaucoma'),
('Hypermetropia'),
('Keratoconus'),
('Leber congetinal amaurosis'),
('Macular hole'),
('Myopia (short sight)'),
('Pemphigoid'),
('Presbyopia'),
('Ptosis'),
('Removal of and eye'),
('Retinal Detachment'),
('Retinal vein occlusion'),
('Squint (strabismus)'),
('Stargardt disease'),
('Styes'),
('Uveal melanoma'),
('Uveitis');


INSERT INTO Admissions (patient_id, admission_date, Eyecondition_id)
VALUES
(1, '2021-11-01 12:53', 1),
(2, '2021-12-03 15:53', 2);

        