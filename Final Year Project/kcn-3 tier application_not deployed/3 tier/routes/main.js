// Route handler for forum web app

module.exports = function (app, HospitalData) {

    // Handle our routes

    // Home page
    app.get('/', function (req, res) {
        res.render('index.ejs', HospitalData)
    });

    app.get('/index', function (req, res) {
        res.render('index.ejs', HospitalData)
    });
    // About page
    app.get('/about', function (req, res) {
        res.render('about.ejs', HospitalData);
    });

    // List patients page
    app.get('/patients', function (req, res) { // Query to select all patients
        let sqlquery = 'SELECT * FROM Patients;'

        // Run the query
        db.query(sqlquery, (err, result) => {
            if (err) {
                res.redirect('./');
            } else {
                
                // Pass results to the EJS page and view it
                let data = Object.assign({}, HospitalData, {patients: result});
                // year_born = patient.dob.getFullYear();
                // var age = currentYear - year_born;
                // age = patient.age;
                // console.log(data)
                res.render('patients.ejs', data);

            }


        });
    });

    // List patients page
    app.get('/paedpatients', function (req, res) { // Query to select all patients
        let sqlquery = 'SELECT * FROM Patients WHERE dob < 23/11/2016'

        // Run the query
        db.query(sqlquery, (err, result) => {
            if (err) {
                res.redirect('./');
            }

            // Pass results to the EJS page and view it
            let data = Object.assign({}, HospitalData, {patients: result});
            console.log(data)
            res.render('paedpatients.ejs', data);
        });
    });


    // Eye Condition page
    app.get('/eyeconditions', function (req, res) { // Query to select all patients
        let sqlquery = 'SELECT * FROM EyeConditions;'

        // Run the query
        db.query(sqlquery, (err, result) => {
            if (err) {
                res.redirect('./');
            }

            // Pass results to the EJS page and view it
            let data = Object.assign({}, HospitalData, {eyeconditions: result});
            console.log(data)
            res.render('eyeconditions.ejs', data);
        });
    });

    // Search for Patients page
    app.get('/search', function (req, res) {
        res.render('search.ejs', HospitalData);
    });

    // Search for Patients form handler
    app.get('/search-result', function (req, res) { // searching in the database
        let term = '%' + req.query.keyword + '%'
        let sqlquery = `SELECT *
                        FROM   Patients
                        WHERE  patient_id LIKE ? OR firstname LIKE ? OR surname LIKE ? OR dob LIKE ? OR sex LIKE ? OR race LIKE ?`

        db.query(sqlquery, [
            term,
            term,
            term,
            term,
            term,
            term
        ], (err, result) => {
            if (err) {
                res.redirect('./');
            }

            let data = Object.assign({}, HospitalData, {patients: result});
            res.render('patients.ejs', data);
        });
    });


    // Add a New Patient page
    app.get('/addpatient', function (req, res) { // Set the initial values for the form
        let initialvalues = {
            firstname: '',
            surname: '',
            dob: '',
            sex: '',
            race: ''
        }

        // Pass the data to the EJS page and view it
        return renderAddNewPatient(res, initialvalues, "")
    });

    // Helper function to
    function renderAddNewPatient(res, initialvalues, errormessage) {
        let data = Object.assign({}, HospitalData, initialvalues, {errormessage: errormessage});
        console.log(data)
        res.render("addpatient.ejs", data);
        return
    }

    // Add a New Patient page form handler
    app.post('/patientadded', function (req, res) {

        let params = [
            req.body.firstname,
            req.body.surname,
            req.body.dob,
            req.body.sex,
            req.body.race
        ];

        sqlquery = "INSERT INTO Patients (firstname, surname, dob, sex, race) VALUES (?,?,?,?,?)";
        db.query(sqlquery, params, (err, result) => {
            if (err) {
                return console.error(err.message)
            } else {


                res.render('patientadded.ejs', HospitalData)
            }
        });
    });


    // Delete page
    app.get('/delete/:patient', function (req, res) {
        console.log(req.params)

        let sql = `DELETE FROM patients
                          WHERE patient_id=?`;

        db.query(sql, [req.params.post], (err, result) => {
            if (err) {
                res.redirect('./');
            }

            res.send('Patient deleted');
        });
    });
}
