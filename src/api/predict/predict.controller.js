const crypto = require('crypto');
const { predictImageSegmentation } = require('../predict/predict.model');
const { storeData, getDataByUserID, getDataByID, deleteDataByID } = require('./predict.service');

const postPredict = async (req, res) => {
    try {
        if (!req.files || req.files.length === 0) {
            return res.status(400).json({ error: true, message: 'No files uploaded' });
        }

        const images = req.files.map(file => file.buffer); 
        const { objectDetectionModel } = req.models; 

        const { results } = await predictImageSegmentation(objectDetectionModel, images);
        // console.log(results)
        const id = crypto.randomUUID();
        const createdAt = new Date().toISOString();

        const structuredResults = results.map(item => ({
            objectDetected: item.class,
            image: `data:image/jpeg;base64,${item.image}`,
        }));

        const data = {
            id: id,
            result: structuredResults,
            createdAt: createdAt,
        };

        res.status(201).json({
            error: false,
            message: 'success',
            data,
        });
    } catch (error) {
        console.error('Error processing images:', error);
        res.status(500).json({ error: true, message: 'Failed to process images' });
    }
};


module.exports = { postPredict };