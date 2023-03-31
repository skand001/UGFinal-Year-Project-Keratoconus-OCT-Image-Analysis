// My Hospital
// A web application to provide Hospital data

// Import the modules we need
var express = require ('express')
var ejs = require('ejs')
var bodyParser= require ('body-parser')
const mysql = require('mysql');

// Create the express application object
const app = express()
const port = 8000
app.use(bodyParser.urlencoded({ extended: true }))

// Define the database connection
const db = mysql.createConnection ({
    host: 'localhost',
    user: 'hospitalapp',
    password: 'h0sp1t4l',
    database: 'eyeHospital'
});
// Connect to the database
db.connect((err) => {
    if (err) {
        throw err;
    }
    console.log('Connected to database');
});
global.db = db;

// Set the directory where static files (css, js, etc) will be
app.use(express.static(__dirname + "/public"));

// Set the directory where Express will pick up HTML files
// __dirname will get the current directory
app.set('views', __dirname + '/views');

// Tell Express that we want to use EJS as the templating engine
app.set('view engine', 'ejs');

// Tells Express how we should process html files
// We want to use EJS's rendering engine
app.engine('html', ejs.renderFile);

// Define our data
var hospitalData = {HospitalName: "Keratoconus Detection"}
var date = new Date();
var currentYear = date.getFullYear();

// Requires the main.js file inside the routes folder passing in the Express app and data as arguments.  All the routes will go in this file

require("./routes/main")(app, hospitalData);

// Start the web app listening
app.listen(port, () => console.log(`Keratoconus Web-Applications is listening on port ${port}!`))
