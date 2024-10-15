const express = require('express');
const multer = require('multer');
const { postPredict } = require('./predict.controller');

const router = express.Router();
const upload = multer({ storage: multer.memoryStorage() }); 

// router.post('/parking-lot', upload.array('video'), postPredict); 
router.post('/parking-lot', upload.array('image'), postPredict); 

// router.get('/history', getPredictionHistories)
module.exports = router;
