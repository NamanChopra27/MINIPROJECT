const express = require('express');
const router = express.Router();

const { login, signup, register, lectureQnA,allPatients,getLectureSummary, getPatient,getText, saveReport, } = require('../controller/authController');

router.post('/login', login);
router.post('/signup', signup);
router.post('/register', register);
router.get('/allPatients', allPatients);
router.get('/getPatient/:id',getPatient);
router.get('/openai/getText',getText);
 
router.post('/saveReport',saveReport);
 // router.get('/getSummary/:id',getSummary);
// router.post('/chat',chat);

// Define routes
router.post('/lecture-summary', getLectureSummary);
router.post('/lecture-qa', lectureQnA);

module.exports = router;