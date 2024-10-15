const ort = require("onnxruntime-node");
const path = require('path');
const tf = require('@tensorflow/tfjs-node');

async function loadObjectDetectionModel() {
    const modelPath = path.resolve(__dirname, '../../models/object-detection/best.onnx');
    const model = await ort.InferenceSession.create(modelPath);
    return model;
}

async function loadRegressionModel() {
    const modelPath = 'file://' + path.resolve(__dirname, '../../models/model-regression/model.json');
    const model = await tf.loadGraphModel(modelPath);
    return model;

}

module.exports = {loadObjectDetectionModel, loadRegressionModel}; 